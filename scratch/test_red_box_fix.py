import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO
import sys
sys.path.insert(0, ".")
import run_video_encroachment as rve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()
yolo_model = YOLO("yolo26n-sem.pt")

cap = cv2.VideoCapture("video/test2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Let's inspect frame 98
cap.set(cv2.CAP_PROP_POS_FRAMES, 98)
ret, frame_bgr = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# YOLO
yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
if yolo_pred.shape != (h, w):
    yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

# SegFormer
inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = seg_model(**inputs)
    logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy()

mask_sw = (pred == 1) | (yolo_pred == 1)
mask_road = (pred == 0) & (~mask_sw)
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18])

(la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)
corridor, pts_l, pts_r, lx_by_y, rx_by_y, _, _ = rve.extract_curved_road_corridor(
    mask_road, mask_sw, veh_mask, h, w, yolo_lines=((la, lb), (ra, rb))
)

# Vehicles
vehicle_ids = [k for k, v in seg_model.config.id2label.items() if v in {"car", "truck", "bus", "motorcycle", "bicycle", "rider", "person"}]
vehicle_mask = np.isin(pred, vehicle_ids).astype(np.uint8) * 255
vehicle_dilated = cv2.dilate(vehicle_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

# Difference
normal_mask = np.isin(pred, [0, 1] + [k for k, v in seg_model.config.id2label.items() if v in rve.CORRIDOR_CLASSES]).astype(np.uint8) * 255
road_dilated = cv2.dilate((pred == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

encroach_raw = (corridor == 255) & (normal_mask == 0) & (road_dilated > 0)
encroach_mask = cv2.morphologyEx(encroach_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
cnts, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Frame 98: found {len(cnts)} raw contours")
sand_cnt = None
for c in cnts:
    area = cv2.contourArea(c)
    bx, by, bw, bh = cv2.boundingRect(c)
    print(f"  cnt bbox: ({bx},{by},{bw},{bh}), area={area}")
    if area > 300:
        sand_cnt = c

# Compare 3 ways to draw:
# 1. Old get_solid_obstacle_mask (stretches from lx to rx)
# 2. Actual contour only
# 3. Solid mask but masked by (~vehicle_mask) so it NEVER covers the vehicles!

# Check what building class (sand pile) has at y in [420..500]
bld_mask = (pred == 2).astype(np.uint8) * 255
print(f"Building pixels in corridor: {np.sum((bld_mask > 0) & (corridor == 255))}")

# Let's visualize
vis1 = frame_bgr.copy()
# Draw old solid mask
dummy_track = {"cnt": sand_cnt, "side": "Trái"}
old_solid = rve.get_solid_obstacle_mask(dummy_track, h, w, lx_by_y, rx_by_y)
vis1[old_solid == 255] = cv2.addWeighted(vis1[old_solid == 255], 0.5, np.full_like(vis1[old_solid == 255], (0, 0, 255)), 0.5, 0)
cv2.imwrite("scratch/vis_f98_old_solid.jpg", vis1)

# Way 2: Actual obstacle contour (plus building class if sand pile is building)
vis2 = frame_bgr.copy()
# Actual contour
c_mask = np.zeros((h, w), dtype=np.uint8)
cv2.drawContours(c_mask, [sand_cnt], -1, 255, -1)
# subtract vehicles
c_mask[vehicle_mask > 0] = 0
vis2[c_mask == 255] = cv2.addWeighted(vis2[c_mask == 255], 0.5, np.full_like(vis2[c_mask == 255], (0, 0, 255)), 0.5, 0)
cnts2, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis2, cnts2, -1, (0, 0, 255), 3, cv2.LINE_AA)
cv2.imwrite("scratch/vis_f98_contour_only.jpg", vis2)

# Way 3: Connect to curb BUT exclude vehicles (never draw on top of vehicle/rider)
vis3 = frame_bgr.copy()
new_solid = old_solid.copy()
new_solid[vehicle_mask > 0] = 0
vis3[new_solid == 255] = cv2.addWeighted(vis3[new_solid == 255], 0.5, np.full_like(vis3[new_solid == 255], (0, 0, 255)), 0.5, 0)
cnts3, _ = cv2.findContours(new_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(vis3, cnts3, -1, (0, 0, 255), 3, cv2.LINE_AA)
cv2.imwrite("scratch/vis_f98_exclude_veh.jpg", vis3)

print("Saved scratch/vis_f98_old_solid.jpg, scratch/vis_f98_contour_only.jpg, scratch/vis_f98_exclude_veh.jpg")
