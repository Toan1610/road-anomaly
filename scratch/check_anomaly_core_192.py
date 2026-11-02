import cv2
import torch
import numpy as np
from run_video_encroachment_mask2former import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 191)
ret, frame = cap.read()
H, W = frame.shape[:2]
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

probs = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, True)
pred = probs.argmax(dim=0).cpu().numpy()
pred_u8 = pred.clip(0, 255).astype(np.uint8)

gpu_morph = GpuMorphology(device)
labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
normal_classes = [0, 1] + [k for k, v in labels.items() if v in CORRIDOR_CLASSES]
LUT_NORMAL = make_class_lut(normal_classes)

normal_mask = (LUT_NORMAL[pred_u8]).astype(np.uint8) * 255
normal_mask = gpu_morph.close(normal_mask, iterations=7)

road_mask_pure = (pred == 0).astype(np.uint8)
road_dilated = gpu_morph.dilate(road_mask_pure, iterations=30)

y_top = int(H * 0.35)
y_bottom = int(H * 0.95)
mask_sw = (pred == 1)
mask_infra = LUT_INFRA[pred_u8].astype(np.uint8)
mask_road = (pred == 0) & (~mask_sw)
veh_mask = LUT_VEHICLE[pred_u8] & (~mask_sw)

(
    legal_corridor_poly,
    pts_left_curve,
    pts_right_curve,
    left_x_by_y,
    right_x_by_y,
    _, _
) = extract_curved_road_corridor(
    mask_road,
    mask_sw,
    veh_mask,
    H,
    W,
    y_top=y_top,
    y_bot=y_bottom,
    alpha=0.15,
    mask_infra=mask_infra,
)

y = 550
print("At Frame 192, y = 550:")
print(f"  legal_corridor_poly nonzeros: {np.where(legal_corridor_poly[y] > 0)[0][[0, -1]] if np.any(legal_corridor_poly[y] > 0) else 'None'}")
print(f"  road nonzeros: {np.where(road_mask_pure[y] > 0)[0][[0, -1]] if np.any(road_mask_pure[y] > 0) else 'None'}")
print(f"  road_dilated nonzeros: {np.where(road_dilated[y] > 0)[0][[0, -1]] if np.any(road_dilated[y] > 0) else 'None'}")
print(f"  normal_mask nonzeros: {np.where(normal_mask[y] > 0)[0][[0, -1]] if np.any(normal_mask[y] > 0) else 'None'}")

horizon_cutoff = max(int(H * 0.50), y_top + 65)
is_road_level = (np.arange(H)[:, None] >= horizon_cutoff)

anomaly_core = (
    (legal_corridor_poly == 255)
    & (normal_mask == 0)
    & is_road_level
    & (road_dilated > 0)
    & (~veh_mask)
)
print(f"  anomaly_core nonzeros: {np.where(anomaly_core[y] > 0)[0][[0, -1]] if np.any(anomaly_core[y] > 0) else 'None'}")

