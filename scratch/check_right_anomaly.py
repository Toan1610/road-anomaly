import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_video_encroachment_mask2former import (
    infer_mask2former_probs,
    extract_curved_road_corridor,
    make_class_lut,
    CORRIDOR_CLASSES,
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

corridor = cv2.imread('scratch/masks_debug/f190_corridor.png', 0)
normal = cv2.imread('scratch/masks_debug/f190_normal.png', 0)
right_anomaly = (corridor == 255) & (normal == 0) & (np.arange(W)[None, :] >= 640)

crop_labels = pred[right_anomaly]
unq, cnts = np.unique(crop_labels, return_counts=True)
for u, c in zip(unq, cnts):
    print(f"Class {u} ({model.config.id2label.get(u, 'unknown')}): {c} pixels ({c/len(crop_labels)*100:.1f}%)")

