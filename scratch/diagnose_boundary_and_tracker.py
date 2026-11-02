import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()
yolo_model = YOLO("yolo26n-sem.pt")

import sys
sys.path.insert(0, ".")
import run_video_encroachment as rve

cap = cv2.VideoCapture("video/test2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video {w}x{h} @ {fps} FPS, {total_frames} frames")

frames_data = {}
f_idx = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if 80 <= f_idx <= 145:
        frames_data[f_idx] = frame_bgr.copy()
    f_idx += 1
cap.release()

print(f"Loaded frames 80..145 ({len(frames_data)} frames)")

# Test current extract_curved_road_corridor vs vehicle-occlusion aware on frame 85
f85 = frames_data[85]
frame_rgb = cv2.cvtColor(f85, cv2.COLOR_BGR2RGB)

# Run SegFormer
inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = seg_model(**inputs)
    logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy()

# Run YOLO
yolo_res = yolo_model(f85, imgsz=(736, 1280), device=0, verbose=False)[0]
yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
if yolo_pred.shape != (h, w):
    yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

mask_sw = (pred == 1) | (yolo_pred == 1)
mask_road = (pred == 0) & (~mask_sw)
# Vehicles: 11: person, 12: rider, 13: car, 14: truck, 15: bus, 17: motorcycle, 18: bicycle
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
veh_dil = cv2.dilate(veh_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

# Check clean road
cnts, _ = cv2.findContours(mask_road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_road = np.zeros((h, w), dtype=np.uint8)
if cnts:
    max_area = max(cv2.contourArea(c) for c in cnts)
    for c in cnts:
        if cv2.contourArea(c) > max(1000.0, max_area * 0.05):
            cv2.drawContours(clean_road, [c], -1, 255, -1)

y_top = int(h * 0.35)
y_bot = int(h * 0.98)

print("Comparing left boundary detection at various y scanlines for Frame 85:")
print(f"{'y':>5} | {'cur_lx (clean_road)':>20} | {'veh_dil[y] range':>20} | {'is_occluded?':>15}")
for y in range(y_top, y_bot, 20):
    xs = np.where(clean_road[y] > 0)[0]
    v_xs = np.where(veh_dil[y] > 0)[0]
    cur_lx = xs[0] if len(xs) > 0 else -1
    v_range = f"[{v_xs[0]}, {v_xs[-1]}]" if len(v_xs) > 0 else "None"
    
    # Occlusion check: Does a vehicle exist at or left of cur_lx + 20?
    is_occ = False
    if len(v_xs) > 0 and cur_lx >= 0:
        # if vehicle starts to the left of road or overlaps the road's left edge
        if v_xs[0] <= cur_lx + 30 and v_xs[-1] >= cur_lx - 30:
            is_occ = True
    print(f"{y:>5} | {cur_lx:>20} | {v_range:>20} | {str(is_occ):>15}")
