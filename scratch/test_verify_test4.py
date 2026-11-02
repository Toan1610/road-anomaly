import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_server_mask2former import (
    extract_curved_road_corridor,
    GpuMorphology,
    make_class_lut,
    CORRIDOR_CLASSES,
    CITYSCAPES_19_LABELS,
    EncroachTracker,
    is_corner_artifact,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

MEAN_GPU = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
STD_GPU = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
pixel_mask = torch.ones((1, 384, 384), device=device, dtype=torch.int64)

gpu_morph = GpuMorphology(device)
road_id = 0
sidewalk_id = 1
normal_classes = [road_id, sidewalk_id] + [k for k, v in CITYSCAPES_19_LABELS.items() if v in CORRIDOR_CLASSES]
LUT_NORMAL = make_class_lut(normal_classes)
LUT_VEHICLE = make_class_lut([11, 12, 13, 14, 15, 16, 17])

cap = cv2.VideoCapture("video/test4.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

tracker = EncroachTracker(fps=fps, min_duration_sec=0.5)

frame_idx = 0
prev_curve_left = None
prev_curve_right = None
t_start = time.perf_counter()

print(f"Testing test4.mp4 for false positives (first 300 frames)...")

while cap.isOpened() and frame_idx < 80:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    H, W = frame_bgr.shape[:2]
    out_height, out_width = H, W
    y_top = int(H * 0.35)
    y_bottom = int(H * 0.95)
    
    t_frame = torch.from_numpy(frame_bgr).to(device, non_blocking=True)
    t_rgb = t_frame[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(torch.float16) / 255.0
    t_resized = F.interpolate(t_rgb, size=(384, 384), mode="bilinear", align_corners=False)
    t_norm = (t_resized - MEAN_GPU) / STD_GPU
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=t_norm, pixel_mask=pixel_mask)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        probs_low = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        probs = F.interpolate(probs_low, size=(H, W), mode="bilinear", align_corners=False)[0]
    
    max_prob, pred_tensor = torch.max(probs, dim=0)
    pred = pred_tensor.cpu().numpy()
    max_prob_np = max_prob.cpu().numpy()
    pred_u8 = pred.clip(0, 255).astype(np.uint8)
    
    mask_sw = (pred == sidewalk_id)
    mask_road = (pred == road_id) & (~mask_sw)
    veh_mask = LUT_VEHICLE[pred_u8] & (~mask_sw)
    mask_infra = (mask_sw | (pred == 4) | (pred == 8) | (pred == 9)).astype(np.uint8)
    
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
        out_height,
        out_width,
        y_top=y_top,
        y_bot=y_bottom,
        prev_left_curve=prev_curve_left,
        prev_right_curve=prev_curve_right,
        alpha=0.25,
        mask_infra=mask_infra,
    )
    
    normal_mask_raw = LUT_NORMAL[pred_u8].copy()
    is_low_conf_bike = (pred_u8 == 18) & (max_prob_np < 0.20)
    normal_mask_raw[is_low_conf_bike] = False
    
    normal_mask = (normal_mask_raw.astype(np.uint8)) * 255
    normal_mask = gpu_morph.close(normal_mask, iterations=7)
    
    road_mask_pure = (pred == road_id).astype(np.uint8)
    road_dilated = gpu_morph.dilate(road_mask_pure, iterations=30)
    
    horizon_cutoff = max(int(out_height * 0.50), y_top + 65)
    is_road_level = (np.arange(out_height)[:, None] >= horizon_cutoff)
    
    is_vehicle = veh_mask
    anomaly_core = (
        (legal_corridor_poly == 255)
        & (normal_mask == 0)
        & is_road_level
        & (road_dilated > 0)
        & (~is_vehicle)
    )
    
    obs_closed = gpu_morph.close(anomaly_core.astype(np.uint8) * 255, iterations=4)
    encroach_mask = gpu_morph.open(obs_closed, iterations=3)
    
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_detections = []
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    vehicle_mask = (LUT_VEHICLE[pred_u8]).astype(np.uint8) * 255
    vehicle_dilated = gpu_morph.dilate(vehicle_mask, iterations=7)
    
    for cnt in sorted_contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        rel_y = max(0.0, min(1.0, (by + bh * 0.5 - y_top) / max(1.0, y_bottom - y_top)))
        dyn_min_area = max(400.0, 600.0 * (0.8 + 0.4 * rel_y))
        if area < dyn_min_area:
            continue
        if is_corner_artifact(cnt, out_width, out_height, y_top, y_bottom):
            continue
        if by <= y_top + 5 and bh <= 45:
            continue
        
        cnt_mask_crop = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(cnt_mask_crop, [cnt - np.array([bx, by])], -1, 1, -1)
        cnt_px = int(np.count_nonzero(cnt_mask_crop))
        if cnt_px <= 0:
            continue
        
        veh_crop = vehicle_dilated[by : by + bh, bx : bx + bw]
        overlap_vehicle = int(np.count_nonzero((cnt_mask_crop == 1) & (veh_crop == 255)))
        if overlap_vehicle / cnt_px > 0.40:
            continue
        
        road_crop = road_dilated[by : by + bh, bx : bx + bw]
        road_contact_px = int(np.count_nonzero((cnt_mask_crop == 1) & (road_crop > 0)))
        if road_contact_px / cnt_px < 0.20:
            continue
            
        candidate_detections.append({
            "cnt": cnt,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "area": float(area),
            "depth": float(bw),
            "side": "Trái",
        })
        
    active_tracks, finished_tracks = tracker.update(candidate_detections, frame_idx)

cap.release()
flushed = tracker.flush_confirmed()
elapsed = time.perf_counter() - t_start
fps_actual = frame_idx / elapsed

print("\n--- TEST4 RESULTS ---")
print(f"Processed: {frame_idx} frames in {elapsed:.2f}s ({fps_actual:.1f} FPS)")
print(f"Confirmed incidents: {len(flushed)}")
for inc in flushed:
    print(f"  Incident ID {inc['track_id']}: Start={inc['start_frame']}, Peak={inc['peak_frame']}, Last={inc['last_seen']}")
