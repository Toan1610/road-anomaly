import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
sys.path.insert(0, ".")
import run_video_encroachment as rve

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

yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
if yolo_pred.shape != (h, w):
    yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

(la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)
print(f"YOLO dynamic lines: Left: a={la:.4f}, b={lb:.2f} | Right: a={ra:.4f}, b={rb:.2f}")

y_top = int(h * 0.35)
y_bot = int(h * 0.98)
print(f"Left line at y_top={y_top}: x={la*y_top+lb:.1f} | at y_bot={y_bot}: x={la*y_bot+lb:.1f}")
print(f"Right line at y_top={y_top}: x={ra*y_top+rb:.1f} | at y_bot={y_bot}: x={ra*y_bot+rb:.1f}")
