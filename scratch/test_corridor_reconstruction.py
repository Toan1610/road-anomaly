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
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

f_idx = 0
frame_bgr = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if f_idx == 85:
        frame_bgr = frame.copy()
        break
    f_idx += 1
cap.release()

frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = seg_model(**inputs)
    logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy()

yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
if yolo_pred.shape != (h, w):
    yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

mask_sw = (pred == 1) | (yolo_pred == 1)
mask_road = (pred == 0) & (~mask_sw)

# Vehicles on road
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
veh_dil = cv2.dilate(veh_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

cnts, _ = cv2.findContours(mask_road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_road = np.zeros((h, w), dtype=np.uint8)
if cnts:
    max_area = max(cv2.contourArea(c) for c in cnts)
    for c in cnts:
        if cv2.contourArea(c) > max(1000.0, max_area * 0.05):
            cv2.drawContours(clean_road, [c], -1, 255, -1)

y_top = int(h * 0.35)
y_bot = int(h * 0.98)
all_ys = np.arange(y_top, y_bot + 1)

(la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)

left_x = np.full(h, -1.0, dtype=np.float32)
right_x = np.full(h, -1.0, dtype=np.float32)

for y in all_ys:
    r_xs = np.where(clean_road[y] > 0)[0]
    v_xs = np.where(veh_dil[y] > 0)[0]
    
    # Left boundary check
    if len(r_xs) > 0:
        cur_lx = float(r_xs[0])
        # Is cur_lx occluded by vehicle?
        # A vehicle is occluding the left edge if vehicle pixels exist at or left of cur_lx
        is_left_occluded = False
        if len(v_xs) > 0:
            if v_xs[0] <= cur_lx + 25:
                is_left_occluded = True
        
        # Also check if cur_lx deviates significantly to the right of YOLO expected line
        exp_lx = la * y + lb
        if cur_lx > exp_lx + 100:
            is_left_occluded = True
            
        if not is_left_occluded:
            left_x[y] = cur_lx
            
    # Right boundary check
    if len(r_xs) > 0:
        cur_rx = float(r_xs[-1])
        is_right_occluded = False
        if len(v_xs) > 0:
            if v_xs[-1] >= cur_rx - 25:
                is_right_occluded = True
        exp_rx = ra * y + rb
        if cur_rx < exp_rx - 100:
            is_right_occluded = True
            
        if not is_right_occluded:
            right_x[y] = cur_rx

valid_l_ys = np.where(left_x >= 0)[0]
valid_l_ys = valid_l_ys[(valid_l_ys >= y_top) & (valid_l_ys <= y_bot)]

valid_r_ys = np.where(right_x >= 0)[0]
valid_r_ys = valid_r_ys[(valid_r_ys >= y_top) & (valid_r_ys <= y_bot)]

print(f"Frame 85: valid left ys count = {len(valid_l_ys)}, valid right ys count = {len(valid_r_ys)}")
print(f"Valid left ys range: {valid_l_ys[0] if len(valid_l_ys) else 'None'} .. {valid_l_ys[-1] if len(valid_l_ys) else 'None'}")
