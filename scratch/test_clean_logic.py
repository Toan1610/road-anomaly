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

# 1. Road mask
mask_road = (pred == 0).astype(np.uint8) * 255

# 2. Moving vehicles (Cityscapes 11..18)
mask_vehicles = np.isin(pred, [11, 12, 13, 14, 15, 16, 17, 18]).astype(np.uint8) * 255
k_veh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
veh_dilated = cv2.dilate(mask_vehicles, k_veh)

# 3. Define the drivable road ROI (from hood to near-horizon)
# Dashcam hood is bottom 5% (y > 680), horizon is y < 410
y_top = int(h * 0.56)  # 403 px
y_bot = int(h * 0.94)  # 676 px

roi = np.zeros((h, w), dtype=np.uint8)
roi[y_top:y_bot, :] = 255

# 4. Obstacle mask M_object:
# Non-road static entities inside ROI (construction material, sand, stalls, building intrusions)
# Exclude moving vehicles and road
mask_obstacle = ((pred != 0) & (pred != 10)).astype(np.uint8) * 255
mask_obstacle[veh_dilated > 0] = 0
mask_obstacle = cv2.bitwise_and(mask_obstacle, roi)

# 5. Drivable Corridor M_road:
# Find road boundaries at lower half (unoccluded near-field road)
clean_rows = []
left_edges = []
right_edges = []
for y in range(int(h * 0.65), y_bot, 4):
    r_idx = np.where(mask_road[y, :] > 0)[0]
    if len(r_idx) > 100:
        clean_rows.append(y)
        left_edges.append(r_idx[0])
        right_edges.append(r_idx[-1])

clean_rows = np.array(clean_rows)
left_edges = np.array(left_edges)
right_edges = np.array(right_edges)

# Linear fit for curbs: x = slope * y + intercept
poly_l = np.polyfit(clean_rows, left_edges, 1)
poly_r = np.polyfit(clean_rows, right_edges, 1)

# Find intersection (vanishing point y)
# poly_l[0]*y + poly_l[1] = poly_r[0]*y + poly_r[1]
denom = poly_l[0] - poly_r[0]
if abs(denom) > 1e-4:
    y_converge = (poly_r[1] - poly_l[1]) / denom
    x_converge = poly_l[0] * y_converge + poly_l[1]
else:
    y_converge = y_top
    x_converge = w // 2

# Actual top of drivable corridor must be below convergence point
effective_ytop = max(y_top, int(y_converge + 20))

corridor_pts_l = []
corridor_pts_r = []
curb_l_xs = np.zeros(h, dtype=np.float32)
curb_r_xs = np.full(h, w - 1, dtype=np.float32)

for y in range(effective_ytop, y_bot):
    xl = poly_l[0] * y + poly_l[1]
    xr = poly_r[0] * y + poly_r[1]
    xl = max(0.0, min(float(w - 1), xl))
    xr = max(xl + 10.0, min(float(w - 1), xr))
    curb_l_xs[y] = xl
    curb_r_xs[y] = xr
    corridor_pts_l.append([xl, y])
    corridor_pts_r.append([xr, y])

M_road = np.zeros((h, w), dtype=np.uint8)
drivable_poly = np.array(corridor_pts_l + corridor_pts_r[::-1], dtype=np.int32)
cv2.fillPoly(M_road, [drivable_poly], 255)

# Also union with actual road pixels in this region
M_road = cv2.bitwise_or(M_road, cv2.bitwise_and(mask_road, roi))

# 6. LOGIC: M_intrusion = M_road ∩ M_object
M_intrusion = cv2.bitwise_and(M_road, mask_obstacle)

# Morphological clean up
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
clean_int = cv2.morphologyEx(M_intrusion, cv2.MORPH_OPEN, k3, iterations=1)
clean_int = cv2.morphologyEx(clean_int, cv2.MORPH_CLOSE, k5, iterations=2)

# Find contours
contours, _ = cv2.findContours(clean_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = frame_bgr.copy()
# Overlay M_road
road_overlay = np.zeros_like(vis)
road_overlay[M_road > 0] = (0, 200, 50)
vis = cv2.addWeighted(vis, 0.75, road_overlay, 0.25, 0)

# Draw curbs
pts_l = np.array([[curb_l_xs[y], y] for y in range(effective_ytop, y_bot)], dtype=np.int32)
pts_r = np.array([[curb_r_xs[y], y] for y in range(effective_ytop, y_bot)], dtype=np.int32)
cv2.polylines(vis, [pts_l], False, (0, 255, 255), 2, cv2.LINE_AA)
cv2.polylines(vis, [pts_r], False, (0, 255, 255), 2, cv2.LINE_AA)

print(f"Vanishing point: ({x_converge:.1f}, {y_converge:.1f}), effective_ytop={effective_ytop}")
intrusions_found = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 300:
        continue
    bx, by, bw, bh = cv2.boundingRect(cnt)
    cx = bx + bw // 2
    cy = by + bh // 2
    side = "Le Trai" if cx < (w * 0.5) else "Le Phai"
    intrusions_found += 1
    print(f"  [{side}] BBox=({bx}, {by}, {bw}, {bh}), Area={area:.0f} px")

    # Draw intrusion mask in bright red
    fill_int = np.zeros_like(vis)
    cv2.drawContours(fill_int, [cnt], -1, (0, 0, 255), -1)
    vis = cv2.addWeighted(vis, 1.0, fill_int, 0.5, 0)

    cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 140, 255), 2)
    cv2.putText(vis, f"LAN CHIEM {side}: {int(area)}px", (bx, max(20, by - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imwrite("scratch/test_pipeline_clean.jpg", vis)
print(f"Total intrusions found: {intrusions_found}. Saved scratch/test_pipeline_clean.jpg")
