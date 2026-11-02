import cv2
import numpy as np

# Let's inspect Frame 140 with:
# 1. Old get_solid_obstacle_mask
# 2. Contour only
# 3. Contour masked by non-vehicle

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
cap.set(cv2.CAP_PROP_POS_FRAMES, 140)
ret, frame_bgr = cap.read()
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
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18])

(la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)
corridor, pts_l, pts_r, lx_by_y, rx_by_y, _, _ = rve.extract_curved_road_corridor(
    mask_road, mask_sw, veh_mask, h, w, yolo_lines=((la, lb), (ra, rb))
)

# On frame 140, the sand pile is visible on the left!
# Why did candidate_dets have 0 on frame 140?
# Let's check what contours exist on frame 140 before filtering!
normal_mask = np.isin(pred, [0, 1] + [k for k, v in seg_model.config.id2label.items() if v in rve.CORRIDOR_CLASSES]).astype(np.uint8) * 255
road_dilated = cv2.dilate((pred == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

encroach_raw = (corridor == 255) & (normal_mask == 0) & (road_dilated > 0)
encroach_mask = cv2.morphologyEx(encroach_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
cnts, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Frame 140 raw contours: {len(cnts)}")
for c in cnts:
    print(f"  bbox={cv2.boundingRect(c)}, area={cv2.contourArea(c)}")
