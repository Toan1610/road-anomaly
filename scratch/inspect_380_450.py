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

print("Scanline 380..450 analysis:")
for y in range(380, 460, 10):
    r_xs = np.where(pred[y] == 0)[0]
    sw_xs = np.where(pred[y] == 1)[0]
    bld_xs = np.where(pred[y] == 2)[0]
    veh_xs = np.where(np.isin(pred[y], [11, 12, 13, 14, 15, 17, 18]))[0]
    print(f"y={y:3d} | road={len(r_xs)} {r_xs[:3] if len(r_xs) else '[]'} | sw={len(sw_xs)} | bld={len(bld_xs)} | veh={len(veh_xs)}")
