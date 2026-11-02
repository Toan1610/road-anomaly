import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

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

# Inspect row 500
y = 500
print(f"Pred classes along y={y}:")
for seg_start in range(0, 700, 50):
    chunk_pred = pred[y, seg_start:seg_start+50]
    chunk_yolo = yolo_pred[y, seg_start:seg_start+50]
    counts_pred = {val: int(np.sum(chunk_pred == val)) for val in np.unique(chunk_pred)}
    counts_yolo = {val: int(np.sum(chunk_yolo == val)) for val in np.unique(chunk_yolo)}
    print(f"x=[{seg_start:3d}..{seg_start+50:3d}] | SegFormer: {counts_pred} | YOLO: {counts_yolo}")
