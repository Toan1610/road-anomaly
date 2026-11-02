import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_video_encroachment_mask2former import (
    infer_mask2former_probs,
    extract_curved_road_corridor,
    GpuMorphology,
    make_class_lut,
    EncroachTracker,
    is_corner_artifact,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
road_id = 0
sidewalk_id = 1
# Moving vehicles LUT
LUT_VEHICLE = make_class_lut([11, 12, 13, 14, 15, 16, 17]) # person, rider, car, truck, bus, train, motorcycle
# Normal drivable surface: road (0) and sidewalk (1)
LUT_NORMAL = make_class_lut([0, 1])

gpu_morph = GpuMorphology(device)
tracker = EncroachTracker(fps=fps, min_duration_sec=0.8)

prev_curve_left = None
prev_curve_right = None

y_top = int(H * 0.35)
y_bottom = int(H * 0.95)

frame_idx = 0
incidents = []

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    prob_t = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, use_fp16=True)
    pred = np.argmax(prob_t.cpu().numpy(), axis=0).astype(np.uint8)
    
    mask_road = (pred == road_id).astype(np.uint8) * 255
    mask_sw = (pred == sidewalk_id).astype(np.uint8) * 255
    veh_mask = (LUT_VEHICLE[pred]).astype(bool)
    
    # Sidewalk only for infra check so sand pile (class 3/wall) is interpolated across
    mask_infra = mask_sw.copy()
    
    (
        legal_corridor_poly,
        pts_left_curve,
        pts_right_curve,
        left_x_by_y,
        right_x_by_y,
        prev_curve_left,
        prev_curve_right,
    ) = extract_curved_road_corridor(
        mask_road,
        mask_sw,
        veh_mask,
        H,
        W,
        y_top=y_top,
        y_bot=y_bottom,
        prev_left_curve=prev_curve_left,
        prev_right_curve=prev_curve_right,
        alpha=0.15,
        mask_infra=mask_infra,
    )
    
    # Normal mask: road and sidewalk
    normal_mask = (LUT_NORMAL[pred]).astype(np.uint8) * 255
    normal_mask = gpu_morph.close(normal_mask, iterations=5)
    
    # Road level cutoff
    horizon_cutoff = max(int(H * 0.50), y_top + 65)
    is_road_level = (np.arange(H)[:, None] >= horizon_cutoff)
    
    # Anomaly core inside the legal corridor
    anomaly_core = (
        (legal_corridor_poly == 255)
        & (normal_mask == 0)
        & is_road_level
        & (~veh_mask)
    )
    
    obs_closed = gpu_morph.close(anomaly_core.astype(np.uint8) * 255, iterations=4)
    encroach_mask = gpu_morph.open(obs_closed, iterations=3)
    
    contours, _ = cv2.findContours(
        encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    road_dilated = gpu_morph.dilate((pred == road_id).astype(np.uint8), iterations=35)
    
    candidate_detections = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        
        rel_y = max(0.0, min(1.0, (by + bh * 0.5 - y_top) / max(1.0, y_bottom - y_top)))
        dyn_min_area = max(400.0, 600.0 * (0.8 + 0.4 * rel_y))
        if area < dyn_min_area:
            continue
        
        if is_corner_artifact(cnt, W, H, y_top, y_bottom):
            continue
        if by <= y_top + 5 and bh <= 45:
            continue
        
        cnt_mask_crop = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(cnt_mask_crop, [cnt - np.array([bx, by])], -1, 1, -1)
        cnt_px = int(np.count_nonzero(cnt_mask_crop))
        if cnt_px <= 0:
            continue
        
        # Must touch the segmented road
        road_crop = road_dilated[by : by + bh, bx : bx + bw]
        road_contact = int(np.count_nonzero((cnt_mask_crop == 1) & (road_crop == 1)))
        if road_contact < 30:
            continue
        
        cnt_side = "Trái" if (bx + bw * 0.35) < (W * 0.5) else "Phải"
        mid_y = int(np.clip(by + bh * 0.5, 0, H - 1))
        if cnt_side == "Trái":
            curb_x = left_x_by_y[mid_y] if left_x_by_y is not None else 0
            if (bx - curb_x) > 45:
                continue
        else:
            curb_x = right_x_by_y[mid_y] if right_x_by_y is not None else (W - 1)
            if (curb_x - (bx + bw)) > 45:
                continue
        
        # Calculate max depth from curb
        pts = cnt.reshape(-1, 2)
        y_hood_cutoff = int(H * 0.90)
        valid_pts = pts[(pts[:, 1] >= y_top) & (pts[:, 1] < y_hood_cutoff)]
        cnt_max_depth = 0.0
        cnt_deepest_pt = None
        if len(valid_pts) > 0 and left_x_by_y is not None and right_x_by_y is not None:
            for pt in valid_pts:
                py = int(np.clip(pt[1], 0, H - 1))
                if cnt_side == "Trái":
                    lx = np.clip(left_x_by_y[py], 0.0, W - 1.0)
                    depth = float(pt[0] - lx)
                else:
                    rx = np.clip(right_x_by_y[py], 0.0, W - 1.0)
                    depth = float(rx - pt[0])
                if depth > cnt_max_depth:
                    cnt_max_depth = depth
                    cnt_deepest_pt = (int(pt[0]), int(pt[1]))
        
        if cnt_max_depth < 40.0:
            continue
        
        candidate_detections.append({
            "cnt": cnt,
            "area": area,
            "depth": cnt_max_depth,
            "side": cnt_side,
            "bbox": (bx, by, bw, bh),
            "yolo_pt": cnt_deepest_pt,
            "seg_pt": cnt_deepest_pt,
        })
    
    active_confirmed, finished_tracks = tracker.update(candidate_detections, frame_idx)
    if candidate_detections or active_confirmed:
        print(f"Frame {frame_idx:03d}: candidates={len(candidate_detections)}, active_confirmed={len(active_confirmed)}")
        for t in active_confirmed:
            print(f"  Track ID {t['id']}: side={t['side']}, duration={t['total_seen']}f, area={t['area']}, depth={t['depth']:.1f}")

confirmed = tracker.flush_confirmed()
print("\n=== FINAL CONFIRMED INCIDENTS ===")
for inc in confirmed:
    print(f"Incident {inc['id']}: peak_frame={inc['peak_frame']}, side={inc['side']}, duration_frames={inc['total_seen']}, max_area={inc['max_area']}, max_depth={inc['max_depth']:.1f}")

