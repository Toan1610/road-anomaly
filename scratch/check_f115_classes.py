import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 115)
ret, frame_bgr = cap.read()
cap.release()

inputs = processor(images=Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = seg_model(**inputs)
    logits = F.interpolate(outputs.logits, size=(720, 1280), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy()

for u, c in zip(*np.unique(pred, return_counts=True)):
    print(f"Class {u:2d} ({seg_model.config.id2label[u]:15s}): {c:6d} pixels")
