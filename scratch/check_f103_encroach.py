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
cap.set(cv2.CAP_PROP_POS_FRAMES, 103)
ret, frame_bgr = cap.read()
cap.release()

inputs = processor(images=Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt")
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

normal_mask = np.isin(pred, [0, 1] + [k for k, v in seg_model.config.id2label.items() if v in rve.CORRIDOR_CLASSES]).astype(np.uint8) * 255
road_dilated = cv2.dilate((pred == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

encroach_raw = (corridor == 255) & (normal_mask == 0) & (road_dilated > 0)

print(f"Total raw encroach pixels: {np.sum(encroach_raw)}")
ys, xs = np.where(encroach_raw)
if len(ys) > 0:
    print(f"y range of encroach_raw: [{ys.min()}, {ys.max()}], x range: [{xs.min()}, {xs.max()}]")

# Check all non-normal pixels in corridor (regardless of road_dilated)
all_anomaly_in_corridor = (corridor == 255) & (normal_mask == 0)
print(f"Total anomaly in corridor: {np.sum(all_anomaly_in_corridor)} pixels")
ys_a, xs_a = np.where(all_anomaly_in_corridor)
if len(ys_a) > 0:
    print(f"y range of all_anomaly_in_corridor: [{ys_a.min()}, {ys_a.max()}], x range: [{xs_a.min()}, {xs_a.max()}]")

# Check where building class is in corridor
bld_in_corridor = (corridor == 255) & (pred == 2)
print(f"Building in corridor: {np.sum(bld_in_corridor)} pixels")
ys_b, xs_b = np.where(bld_in_corridor)
if len(ys_b) > 0:
    print(f"y range of building in corridor: [{ys_b.min()}, {ys_b.max()}], x range: [{xs_b.min()}, {xs_b.max()}]")
