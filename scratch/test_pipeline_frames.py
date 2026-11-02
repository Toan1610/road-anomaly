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
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

y_top = int(h * 0.35)
y_bottom = int(h * 0.98)

print("Testing boundary reconstruction across frames 85, 95, 105, 115, 125, 135, 145, 199...")

frames_to_test = [85, 95, 105, 115, 125, 135, 145, 199]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
f_idx = 0
frames = {}
while cap.isOpened() and len(frames) < len(frames_to_test):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if f_idx in frames_to_test:
        frames[f_idx] = frame_bgr.copy()
    f_idx += 1
cap.release()

for fid in frames_to_test:
    frame_bgr = frames[fid]
    yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
    yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)
    (la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)
    print(f"Frame {fid:3d} YOLO lines: Left: a={la:7.4f}, b={lb:7.2f} (at y=500: {la*500+lb:5.1f}) | Right: a={ra:7.4f}, b={rb:7.2f} (at y=500: {ra*500+rb:5.1f})")
