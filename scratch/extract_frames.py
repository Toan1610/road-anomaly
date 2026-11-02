import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models
print("Loading SegFormer...")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()

print("Loading YOLO semantic...")
yolo_model = YOLO("yolo26n-sem.pt")

cap = cv2.VideoCapture("video/test2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video {w}x{h} @ {fps} FPS, {total_frames} frames")

frames_to_check = [85, 95, 100, 110, 115, 120, 130, 140, 199]
saved_frames = {}

f_idx = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if f_idx in frames_to_check:
        saved_frames[f_idx] = frame_bgr.copy()
    f_idx += 1
cap.release()

print(f"Extracted frames: {list(saved_frames.keys())}")
