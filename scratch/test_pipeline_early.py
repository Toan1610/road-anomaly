import cv2
import torch
import numpy as np
import torch.nn.functional as F
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_server_mask2former import (
    extract_curved_road_corridor,
    GpuMorphology,
    make_class_lut,
    CORRIDOR_CLASSES,
    CITYSCAPES_19_LABELS,
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
TRAFFIC_CLASSES = set([11, 12, 13, 14, 15, 16, 17, 18])

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
prev_curve_left = None
prev_curve_right = None

print("Checking detection on test2.mp4:")
detected_frames = []

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    H, W = frame_bgr.shape[:2]
    out_height, out_width = H, W
    y_top = int(H * 0.35)
    y_bottom = int(H * 0.95)
    
    # GPU Tensor inference
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
    
    # Normal mask with low-confidence anomaly handling:
    # If a pixel is in TRAFFIC_CLASSES, require max_prob >= 0.15 to be considered normal traffic.
    # Otherwise, it's an out-of-distribution anomaly!
    is_traffic = np.isin(pred_u8, list(TRAFFIC_CLASSES))
    is_confident_traffic = is_traffic & (max_prob_np >= 0.15)
    is_base_normal = (pred == road_id) | (pred == sidewalk_id) | (pred == 6) | (pred == 7)
    
    normal_mask_np = (is_base_normal | is_confident_traffic).astype(np.uint8) * 255
    normal_mask = gpu_morph.close(normal_mask_np, iterations=7)
    
    road_mask_pure = (pred == road_id).astype(np.uint8)
    road_dilated = gpu_morph.dilate(road_mask_pure, iterations=30)
    
    horizon_cutoff = max(int(out_height * 0.50), y_top + 65)
    is_road_level = (np.arange(out_height)[:, None] >= horizon_cutoff)
    
    anomaly_core = (
        (legal_corridor_poly == 255)
        & (normal_mask == 0)
        & is_road_level
        & (road_dilated > 0)
        & (~veh_mask)
    )
    
    obs_closed = gpu_morph.close(anomaly_core.astype(np.uint8) * 255, iterations=4)
    encroach_mask = gpu_morph.open(obs_closed, iterations=3)
    
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 600:
            cands.append(area)
            
    if len(cands) > 0:
        detected_frames.append((frame_idx, max(cands)))
        if frame_idx % 10 == 0 or frame_idx in [155, 160, 165, 170, 175, 180, 185, 190]:
            print(f"  Frame {frame_idx:03d}: DETECTED obstacle! Max area = {max(cands):.0f} px")

cap.release()
print(f"Total detected frames: {len(detected_frames)}")
if detected_frames:
    print(f"First detected frame: {detected_frames[0][0]} with area {detected_frames[0][1]:.0f}")
    print(f"Last detected frame: {detected_frames[-1][0]} with area {detected_frames[-1][1]:.0f}")
