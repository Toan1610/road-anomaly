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

# Mask definitions
mask_road = (pred == 0).astype(np.uint8) * 255
# Vehicles: 11:person, 12:rider, 13:car, 14:truck, 15:bus, 17:motorcycle, 18:bicycle
mask_vehicles = np.isin(pred, [11, 12, 13, 14, 15, 16, 17, 18]).astype(np.uint8) * 255
k_veh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
veh_dilated = cv2.dilate(mask_vehicles, k_veh)

# Obstacle candidates: non-road, non-sky, non-moving-vehicle
# Classes 1 (sidewalk protruding), 2 (building/material), 3 (wall), 4 (fence), 5 (pole), 7 (sign), 8 (vegetation), 9 (terrain)
mask_nonroad = ((pred != 0) & (pred != 10)).astype(np.uint8) * 255
mask_obstacle = mask_nonroad.copy()
mask_obstacle[veh_dilated > 0] = 0

# Limit ROI to drivable zone in camera view (y between 0.48h and 0.94h)
y_top = int(h * 0.48)
y_bot = int(h * 0.94)

roi_mask = np.zeros((h, w), dtype=np.uint8)
roi_mask[y_top:y_bot, :] = 255
mask_obstacle = cv2.bitwise_and(mask_obstacle, roi_mask)

# 1. Reconstructing Ideal Road Boundary M_road:
# We determine the expected left and right curb lines.
# In frame 105, road is clear near bottom (y > 600) and in center.
# Let us find curb points:
l_pts = []
r_pts = []
for y in range(int(h * 0.58), y_bot, 4):
    road_xs = np.where(mask_road[y, :] > 0)[0]
    if len(road_xs) > 20:
        l_pts.append((y, road_xs[0]))
        r_pts.append((y, road_xs[-1]))

l_pts = np.array(l_pts)
r_pts = np.array(r_pts)

# Fit linear polynomial to bottom road edges
p_left = np.polyfit(l_pts[:, 0], l_pts[:, 1], 1)
p_right = np.polyfit(r_pts[:, 0], r_pts[:, 1], 1)

# Generate ideal corridor mask M_road
M_road = np.zeros((h, w), dtype=np.uint8)
curb_l_xs = np.zeros(h, dtype=np.float32)
curb_r_xs = np.zeros(h, dtype=np.float32)
poly_left_pts = []
poly_right_pts = []

for y in range(y_top, y_bot):
    xl = p_left[0] * y + p_left[1]
    xr = p_right[0] * y + p_right[1]
    # Bound within screen
    xl = max(0, min(w - 1, xl))
    xr = max(0, min(w - 1, xr))
    curb_l_xs[y] = xl
    curb_r_xs[y] = xr
    poly_left_pts.append([xl, y])
    poly_right_pts.append([xr, y])

road_polygon = np.array(poly_left_pts + poly_right_pts[::-1], dtype=np.int32)
cv2.fillPoly(M_road, [road_polygon], 255)

# Also ensure any actual road pixel is part of M_road
M_road = cv2.bitwise_or(M_road, cv2.bitwise_and(mask_road, roi_mask))

# 2. Logic: M_intrusion = M_road ∩ M_object
M_intrusion = cv2.bitwise_and(M_road, mask_obstacle)

# Morphological cleanup
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
clean_intrusion = cv2.morphologyEx(M_intrusion, cv2.MORPH_OPEN, k3, iterations=1)
clean_intrusion = cv2.morphologyEx(clean_intrusion, cv2.MORPH_CLOSE, k3, iterations=2)

# Find contours
contours, _ = cv2.findContours(clean_intrusion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} intrusion contours:")

vis = frame_bgr.copy()
# Overlay M_road (Green)
green_layer = np.zeros_like(vis)
green_layer[M_road > 0] = (0, 200, 50)
vis = cv2.addWeighted(vis, 0.75, green_layer, 0.25, 0)

# Draw ideal curb lines (Cyan)
pts_l = np.array([[curb_l_xs[y], y] for y in range(y_top, y_bot)], dtype=np.int32)
pts_r = np.array([[curb_r_xs[y], y] for y in range(y_top, y_bot)], dtype=np.int32)
cv2.polylines(vis, [pts_l], False, (0, 255, 255), 2, cv2.LINE_AA)
cv2.polylines(vis, [pts_r], False, (0, 255, 255), 2, cv2.LINE_AA)

# Overlay intrusions
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 400:
        continue
    bx, by, bw, bh = cv2.boundingRect(cnt)
    print(f"  Contour bbox=({bx}, {by}, {bw}, {bh}), area={area:.0f} px")
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    cv2.drawContours(vis, [cnt], -1, (0, 50, 255), 2)
    cv2.putText(vis, f"Lan chiem: {int(area)}px", (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite("scratch/test_pipeline_result.jpg", vis)
print("Saved scratch/test_pipeline_result.jpg")
