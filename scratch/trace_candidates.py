import cv2
import numpy as np
import torch
import sys, os

PIDNET_ROOT = os.path.abspath("PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)
    sys.path.insert(0, os.path.join(PIDNET_ROOT, "tools"))

from ultralytics import YOLO
import models as pidnet_models

# Import functions from run_video_encroachment
sys.path.insert(0, os.path.abspath("."))
from run_video_encroachment import (
    load_pidnet_model,
    fit_dynamic_road_lines_yolo,
    extract_curved_road_corridor,
    compute_corridor_confidence,
    EncroachTracker,
    is_corner_artifact,
    CORRIDOR_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("Road_Segmentation_Model/yolo26n-sem.pt")
pidnet_model = load_pidnet_model("pidnet-s", "PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt", device, True)

cap = cv2.VideoCapture("video/test4.mp4")

# Let's inspect frames 2025 to 2048
tracker = EncroachTracker(fps=30, min_duration_sec=0.5)

# We can run directly on frames 2020-2048
prev_yolo_l = None
prev_yolo_r = None
prev_curve_left = None
prev_curve_right = None

for f_idx in range(2025, 2045):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret, frame_bgr = cap.read()
    h, w = frame_bgr.shape[:2]
    
    # YOLO
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred_raw = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred_raw.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        yolo_pred = yolo_pred_raw
        
    # PIDNet
    inp = cv2.resize(frame_bgr, (2048, 1024))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device).half()
    with torch.no_grad():
        out = pidnet_model(inp)
        if isinstance(out, (list, tuple)):
            out = out[1]
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        pred = out.argmax(dim=1)[0].cpu().numpy()
        
    mask_sw = (pred == 1) | (yolo_pred == 1)
    mask_road = (pred == 0) & (~mask_sw)
    veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
    
    (la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(
        yolo_pred, h, w, road_id=0, sidewalk_id=1,
        prev_left=prev_yolo_l, prev_right=prev_yolo_r, alpha=0.15
    )
    prev_yolo_l, prev_yolo_r = (la, lb), (ra, rb)
    
    y_top = int(h * 0.35)
    y_bottom = int(h * 0.98)
    
    (
        legal_corridor_poly,
        pts_left_curve,
        pts_right_curve,
        left_x_by_y,
        right_x_by_y,
        prev_curve_left,
        prev_curve_right,
    ) = extract_curved_road_corridor(
        mask_road, mask_sw, veh_mask, h, w,
        y_top=y_top, y_bot=y_bottom,
        prev_left_curve=prev_curve_left,
        prev_right_curve=prev_curve_right,
        alpha=0.15,
        yolo_lines=((la, lb), (ra, rb)),
    )
    
    # Check normal_mask, anomaly_core
    normal_classes = [0, 1] + [k for k, v in {0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}.items() if v in CORRIDOR_CLASSES]
    normal_mask = np.isin(pred, normal_classes).astype(np.uint8) * 255
    normal_mask = cv2.morphologyEx(normal_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    
    road_mask_pure = (pred == 0).astype(np.uint8)
    road_dilated = cv2.dilate(road_mask_pure, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    
    horizon_cutoff = max(int(h * 0.50), y_top + 65)
    is_road_level = (np.arange(h)[:, None] >= horizon_cutoff)
    is_vehicle = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
    
    anomaly_core = (
        (legal_corridor_poly == 255)
        & (normal_mask == 0)
        & is_road_level
        & (road_dilated > 0)
        & (~is_vehicle)
    )
    
    obs_closed = cv2.morphologyEx(
        anomaly_core.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    encroach_mask = cv2.morphologyEx(
        obs_closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if area < 400: continue
        cnt_side = "Trái" if (bx + bw * 0.35) < (w * 0.5) else "Phải"
        mid_y = int(np.clip(by + bh * 0.5, 0, h - 1))
        curb_x = right_x_by_y[mid_y] if cnt_side == "Phải" else left_x_by_y[mid_y]
        dist = curb_x - (bx + bw) if cnt_side == "Phải" else bx - curb_x
        candidates.append({"bbox": (bx, by, bw, bh), "area": area, "side": cnt_side, "curb_dist": dist})
        
    print(f"Frame {f_idx}: {len(candidates)} candidates -> {candidates}")

cap.release()
