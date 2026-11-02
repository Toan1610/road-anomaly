import cv2
import numpy as np
import os, sys

PIDNET_ROOT = os.path.abspath("PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)
    sys.path.insert(0, os.path.join(PIDNET_ROOT, "tools"))
sys.path.insert(0, os.path.abspath("."))

from ultralytics import YOLO
from run_video_encroachment import fit_dynamic_road_lines_yolo

yolo_model = YOLO("Road_Segmentation_Model/yolo26n-sem.pt")
cap = cv2.VideoCapture("video/test4.mp4")

for f_idx in [322, 986, 1786, 2039]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret, frame_bgr = cap.read()
    h, w = frame_bgr.shape[:2]
    
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred_raw = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred_raw.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        yolo_pred = yolo_pred_raw
        
    (la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(
        yolo_pred, h, w, road_id=0, sidewalk_id=1,
        prev_left=None, prev_right=None, alpha=0.15
    )
    y_bot = int(h * 0.98)
    y_top = int(h * 0.35)
    print(f"Frame {f_idx}:")
    print(f"  YOLO Left line at top={la*y_top+lb:.1f}, bot={la*y_bot+lb:.1f} (la={la:.3f}, lb={lb:.1f})")
    print(f"  YOLO Right line at top={ra*y_top+rb:.1f}, bot={ra*y_bot+rb:.1f} (ra={ra:.3f}, rb={rb:.1f})")

cap.release()
