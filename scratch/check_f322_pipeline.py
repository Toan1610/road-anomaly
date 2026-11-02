import cv2
import numpy as np
import torch
import sys, os

PIDNET_ROOT = os.path.abspath("PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)
    sys.path.insert(0, os.path.join(PIDNET_ROOT, "tools"))
sys.path.insert(0, os.path.abspath("."))

from ultralytics import YOLO
from run_video_encroachment import (
    load_pidnet_model,
    fit_dynamic_road_lines_yolo,
    extract_curved_road_corridor,
    infer_pidnet_probs,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("Road_Segmentation_Model/yolo26n-sem.pt")
pidnet_model = load_pidnet_model("pidnet-s", "PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt", device, True)

cap = cv2.VideoCapture("video/test4.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 321)
ret, frame_bgr = cap.read()
h, w = frame_bgr.shape[:2]

# 1. YOLO
yolo_h = max(32, (h + 31) // 32 * 32)
yolo_w = max(32, (w + 31) // 32 * 32)
yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
yp = yolo_res.semantic_mask.data.cpu().numpy()
if yp.shape != (h, w): yp = cv2.resize(yp, (w, h), interpolation=cv2.INTER_NEAREST)

# 2. PIDNet
probs = infer_pidnet_probs(pidnet_model, frame_bgr, h, w, device, True)
pred = probs.argmax(dim=0).cpu().numpy()

mask_sw = (pred == 1) | (yp == 1)
mask_road = (pred == 0) & (~mask_sw)
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)

(la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(yp, h, w, road_id=0, sidewalk_id=1, alpha=0.15)
y_top = int(h * 0.35)
y_bot = int(h * 0.98)

corr_poly, pts_l, pts_r, lx, rx, cl, cr = extract_curved_road_corridor(
    mask_road, mask_sw, veh_mask, h, w, y_top, y_bot, alpha=0.15, yolo_lines=((la, lb), (ra, rb))
)

print(f"Frame 322:")
print(f"YOLO line at y=817: {ra*817+rb:.1f}, at y=982: {ra*982+rb:.1f}")
print(f"rx at y=817: {rx[817]:.1f}, at y=982: {rx[982]:.1f}")

# Look at bbox [1636, 817, 135, 165]
bx, by, bw, bh = 1636, 817, 135, 165
print(f"Candidate bbox x: [{bx}, {bx+bw}], y: [{by}, {by+bh}]")
print(f"YOLO classes in bbox: {np.unique(yp[by:by+bh, bx:bx+bw], return_counts=True)}")
print(f"PIDNet classes in bbox: {np.unique(pred[by:by+bh, bx:bx+bw], return_counts=True)}")
print(f"mask_road sum in bbox: {np.sum(mask_road[by:by+bh, bx:bx+bw])}")
print(f"mask_sw sum in bbox: {np.sum(mask_sw[by:by+bh, bx:bx+bw])}")
print(f"corr_poly sum in bbox: {np.sum(corr_poly[by:by+bh, bx:bx+bw] == 255)}")

cap.release()
