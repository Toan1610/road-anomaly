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

# Inspect sand pile area: x in [350, 550], y in [420, 550]
# What are the classes in SegFormer and YOLO?
print("Frame 103 analysis in sand pile region (x: 350..550, y: 420..560):")
print(f"{'y':>4} | {'SegFormer dominant classes':>35} | {'YOLO dominant classes':>35}")
for y in range(420, 560, 10):
    sf_row = pred[y, 350:550]
    yo_row = yolo_pred[y, 350:550]
    sf_u, sf_c = np.unique(sf_row, return_counts=True)
    yo_u, yo_c = np.unique(yo_row, return_counts=True)
    sf_str = ", ".join([f"{seg_model.config.id2label[u]}({c})" for u, c in zip(sf_u, sf_c) if c > 10])
    yo_str = ", ".join([f"{u}({c})" for u, c in zip(yo_u, yo_c) if c > 10])
    print(f"{y:>4} | {sf_str:>35} | {yo_str:>35}")

# Save crop of the actual RGB frame in this area
crop_rgb = frame_bgr[420:560, 350:550]
cv2.imwrite("scratch/crop_sand_f103.jpg", crop_rgb)
print("Saved scratch/crop_sand_f103.jpg")
