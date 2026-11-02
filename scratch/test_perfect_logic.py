import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
hf_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
proc = SegformerImageProcessor.from_pretrained(hf_name)
model = SegformerForSemanticSegmentation.from_pretrained(hf_name).to(device).eval()

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 105)
ret, frame_bgr = cap.read()
h, w = frame_bgr.shape[:2]

inputs = proc(images=Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs)
pred = torch.nn.functional.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False).argmax(dim=1)[0].cpu().numpy()

# 1. Mask road & vehicles
mask_road = (pred == 0).astype(np.uint8) * 255
mask_vehicles = np.isin(pred, [11, 12, 13, 14, 15, 16, 17, 18]).astype(np.uint8) * 255
k_veh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
veh_dilated = cv2.dilate(mask_vehicles, k_veh)

# 2. Left and Right road boundaries from curb/sidewalk
# In Cityscapes: 1 is sidewalk, 2 is building, etc.
# Road corridor polygon:
# For test2.mp4, the street has curbs on left and right:
# Vanishing point around (640, 390)
# Bottom left curb around (40, 680), Bottom right curb around (1240, 680)
# Mid left curb around (420, 420), Mid right curb around (860, 420)
# We can estimate curb dynamically:
# Bottom region (y in [600, 680]) gives reliable bottom points
y_bot = int(h * 0.94)
y_top = int(h * 0.58)

# Find uncontaminated curb points at bottom:
left_samples = []
right_samples = []
for y in range(int(h * 0.65), y_bot, 2):
    # Find where sidewalk ends or road begins
    sw_l = np.where(pred[y, :w//2] == 1)[0]
    rd_l = np.where(pred[y, :w//2] == 0)[0]
    if len(sw_l) > 0 and sw_l[-1] < 150:
        left_samples.append((y, sw_l[-1]))
    elif len(rd_l) > 0 and rd_l[0] < 150:
        left_samples.append((y, rd_l[0]))

    rd_r = np.where(pred[y, w//2:] == 0)[0]
    if len(rd_r) > 0:
        right_samples.append((y, w//2 + rd_r[-1]))

# Default perspective road corridor if few samples
if len(left_samples) >= 5:
    l_arr = np.array(left_samples)
    poly_l = np.polyfit(l_arr[:, 0], l_arr[:, 1], 1)
else:
    poly_l = np.array([-1.5, 1080])

if len(right_samples) >= 5:
    r_arr = np.array(right_samples)
    poly_r = np.polyfit(r_arr[:, 0], r_arr[:, 1], 1)
else:
    poly_r = np.array([1.5, 200])

# Generate corridor mask M_road
M_road = np.zeros((h, w), dtype=np.uint8)
pts_l = []
pts_r = []
curb_l = np.zeros(h, dtype=np.float32)
curb_r = np.full(h, w - 1, dtype=np.float32)

for y in range(y_top, y_bot):
    xl = poly_l[0] * y + poly_l[1]
    xr = poly_r[0] * y + poly_r[1]
    # Keep left curb strictly to the left of vanishing point
    xl = max(0.0, min(float(w * 0.45), xl))
    xr = max(float(w * 0.55), min(float(w - 1), xr))
    curb_l[y] = xl
    curb_r[y] = xr
    pts_l.append([xl, y])
    pts_r.append([xr, y])

road_poly = np.array(pts_l + pts_r[::-1], dtype=np.int32)
cv2.fillPoly(M_road, [road_poly], 255)

# Also ensure any actual road pixel in this height range is included
roi_height = np.zeros((h, w), dtype=np.uint8)
roi_height[y_top:y_bot, :] = 255
M_road = cv2.bitwise_or(M_road, cv2.bitwise_and(mask_road, roi_height))

# 3. M_object: All static anomalies (sand/material/stalls/etc.)
# Classes: building (2), wall (3), fence (4), pole (5), sign (7), vegetation (8), terrain (9)
# AND sidewalk (1) if it protrudes into M_road!
anomaly_ids = [1, 2, 3, 4, 5, 7, 8, 9]
M_object = np.isin(pred, anomaly_ids).astype(np.uint8) * 255
# Remove moving vehicles
M_object[veh_dilated > 0] = 0
M_object = cv2.bitwise_and(M_object, roi_height)

# 4. LOGIC: M_intrusion = M_road ∩ M_object
M_intrusion = cv2.bitwise_and(M_road, M_object)

# Morphological clean up
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
clean_int = cv2.morphologyEx(M_intrusion, cv2.MORPH_OPEN, k3, iterations=1)
clean_int = cv2.morphologyEx(clean_int, cv2.MORPH_CLOSE, k5, iterations=2)

# Find contours
contours, _ = cv2.findContours(clean_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = frame_bgr.copy()
# Overlay road (Green)
green = np.zeros_like(vis)
green[M_road > 0] = (0, 210, 80)
vis = cv2.addWeighted(vis, 0.75, green, 0.25, 0)

# Draw curbs (Yellow)
cv2.polylines(vis, [np.array(pts_l, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
cv2.polylines(vis, [np.array(pts_r, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)

print(f"Total contours found: {len(contours)}")
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    bx, by, bw, bh = cv2.boundingRect(cnt)
    cx = bx + bw // 2
    cy = by + bh // 2
    side = "Le Trai" if cx < w // 2 else "Le Phai"

    # Intrusion depth into road
    curb_x = curb_l[cy] if side == "Le Trai" else curb_r[cy]
    depth_px = (bx + bw) - curb_x if side == "Le Trai" else curb_x - bx
    road_width = max(1.0, curb_r[cy] - curb_l[cy])
    depth_pct = min(100.0, max(0.0, depth_px / road_width * 100.0))

    print(f"  [{side}] bbox=({bx}, {by}, {bw}, {bh}), area={area:.0f} px, depth={depth_px:.1f}px ({depth_pct:.1f}%)")

    # Draw intrusion in bright red
    fill_int = np.zeros_like(vis)
    cv2.drawContours(fill_int, [cnt], -1, (0, 0, 255), -1)
    vis = cv2.addWeighted(vis, 1.0, fill_int, 0.55, 0)

    cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 140, 255), 2)

    label = f"LAN CHIEM {side}: {depth_px:.0f}px ({depth_pct:.0f}%) | {int(area)}px"
    cv2.putText(vis, label, (bx, max(22, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imwrite("scratch/perfect_intrusion.jpg", vis)
print("Saved scratch/perfect_intrusion.jpg")
