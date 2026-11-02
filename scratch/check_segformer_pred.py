import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from run_video_encroachment import CITYSCAPES_19_LABELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(m_name)
model = SegformerForSemanticSegmentation.from_pretrained(m_name).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
for _ in range(190): ret, frame_bgr = cap.read()
H, W = frame_bgr.shape[:2]
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

image = Image.fromarray(frame_rgb)
inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
    logits = torch.nn.functional.interpolate(outputs.logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].cpu().numpy()

crop = pred[480:680, 0:300]
unq, cnts = np.unique(crop, return_counts=True)
print("SegFormer predictions on sand pile area frame 190:")
for u, c in zip(unq, cnts):
    print(f"  Class {u} ({CITYSCAPES_19_LABELS[u]}): {c} ({c/crop.size*100:.1f}%)")

