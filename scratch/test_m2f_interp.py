import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_video_encroachment_mask2former import (
    infer_mask2former_probs,
    extract_curved_road_corridor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
for _ in range(190): ret, frame_bgr = cap.read()
H, W = frame_bgr.shape[:2]
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

prob_t = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, use_fp16=True)
pred = np.argmax(prob_t.cpu().numpy(), axis=0).astype(np.uint8)

mask_road = (pred == 0).astype(np.uint8) * 255
mask_sw = (pred == 1).astype(np.uint8) * 255
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 16, 17])

# Test with check_infra = mask_sw (sidewalk only, not wall/fence)
mask_infra = mask_sw.copy()

y_top = int(H * 0.35)
y_bottom = int(H * 0.95)

(
    corridor_poly,
    pts_left,
    pts_right,
    left_x,
    right_x,
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

y = 553
print(f"At y={y}:")
print(f"  left_x[y]={left_x[y]:.1f}")
print(f"  road nonzeros x range: {np.where(mask_road[y] > 0)[0][[0, -1]]}")
print(f"  Difference (obstacle width) at y=553: {np.where(mask_road[y] > 0)[0][0] - left_x[y]:.1f} px")

