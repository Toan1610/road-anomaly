import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_video_encroachment_mask2former import (
    infer_mask2former_probs,
    extract_curved_road_corridor,
    GpuMorphology,
    make_class_lut,
    CITYSCAPES_19_LABELS,
    CORRIDOR_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
road_id = 0
sidewalk_id = 1
normal_classes = [road_id, sidewalk_id] + [k for k, v in labels.items() if v in CORRIDOR_CLASSES]
LUT_NORMAL = make_class_lut(normal_classes)
LUT_VEHICLE = make_class_lut([11, 12, 13, 14, 15, 16, 17])
gpu_morph = GpuMorphology(device)

prev_curve_left = None
prev_curve_right = None

out_dir = Path("scratch/masks_debug")
out_dir.mkdir(parents=True, exist_ok=True)

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    prob_t = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, use_fp16=True)
    prob_np = prob_t.cpu().numpy()
    pred = np.argmax(prob_np, axis=0).astype(np.uint8)
    pred_u8 = pred
    
    mask_road = (pred == road_id).astype(np.uint8) * 255
    mask_sw = (pred == sidewalk_id).astype(np.uint8) * 255
    veh_mask = (LUT_VEHICLE[pred_u8]).astype(bool)
    mask_infra = (pred == 2) | (pred == 3) | (pred == 4) | (pred == 8)
    
    y_top = int(H * 0.35)
    y_bottom = int(H * 0.95)
    
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
        yolo_lines=None,
        mask_infra=mask_infra,
    )
    
    if frame_idx in (184, 190, 192):
        normal_mask = (LUT_NORMAL[pred_u8]).astype(np.uint8) * 255
        normal_mask = gpu_morph.close(normal_mask, iterations=7)
        
        road_mask_pure = (pred == road_id).astype(np.uint8)
        road_dilated = gpu_morph.dilate(road_mask_pure, iterations=25)
        
        horizon_cutoff = max(int(H * 0.50), y_top + 65)
        is_road_level = (np.arange(H)[:, None] >= horizon_cutoff)
        
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
        
        cv2.imwrite(str(out_dir / f"f{frame_idx}_road.png"), mask_road)
        cv2.imwrite(str(out_dir / f"f{frame_idx}_corridor.png"), legal_corridor_poly)
        cv2.imwrite(str(out_dir / f"f{frame_idx}_normal.png"), normal_mask)
        cv2.imwrite(str(out_dir / f"f{frame_idx}_road_dilated.png"), road_dilated * 255)
        cv2.imwrite(str(out_dir / f"f{frame_idx}_anomaly_core.png"), (anomaly_core * 255).astype(np.uint8))
        cv2.imwrite(str(out_dir / f"f{frame_idx}_encroach_mask.png"), encroach_mask)
        print(f"Saved debug masks for frame {frame_idx}")

cap.release()
