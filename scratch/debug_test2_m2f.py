import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_video_encroachment_mask2former import (
    infer_mask2former_probs,
    extract_curved_road_corridor,
    get_solid_obstacle_mask,
    CITYSCAPES_19_LABELS,
    LUT_VEHICLE,
    LUT_INFRA_CROP2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if 182 <= frame_idx <= 194:
        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prob_19_t = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, use_fp16=True)
        prob_19 = prob_19_t.squeeze(0).cpu().numpy()
        pred_label = np.argmax(prob_19, axis=0).astype(np.uint8)
        road_mask = (pred_label == 0).astype(np.uint8) * 255
        
        # Check what classes are in the left region at y=480-650, x=0-300 (where sand pile is)
        crop_labels = pred_label[480:650, 0:300]
        unique, counts = np.unique(crop_labels, return_counts=True)
        class_dist = {CITYSCAPES_19_LABELS[u]: c for u, c in zip(unique, counts) if u < 19}
        
        # Curved corridor extraction
        pts_left, pts_right, left_by_y, right_by_y, _, _ = extract_curved_road_corridor(
            road_mask, min_y_ratio=0.35, max_y_ratio=0.95
        )
        
        # Solid obstacle mask
        solid_mask, debug_meta = get_solid_obstacle_mask(
            prob_19, road_mask, left_by_y, right_by_y,
            min_area=600, min_depth=40.0
        )
        
        print(f"Frame {frame_idx}: sand region classes: {class_dist}")
        print(f"  left_by_y at 550: {left_by_y.get(550, None)}")
        print(f"  debug_meta candidates: {len(debug_meta.get('candidates', []))}")
        for c in debug_meta.get('candidates', []):
            print(f"    cand: area={c['area']}, depth={c['depth']:.1f}, side={c['side']}, bbox={c['bbox']}")

cap.release()
