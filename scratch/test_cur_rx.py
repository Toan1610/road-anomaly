import cv2
import torch
import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from run_video_encroachment_mask2former import infer_mask2former_probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m2f_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(m2f_name, local_files_only=True)
m2f = Mask2FormerForUniversalSegmentation.from_pretrained(m2f_name, local_files_only=True).to(device)
m2f.eval()

cap = cv2.VideoCapture("video/test2.mp4")
for _ in range(190): ret, frame_bgr = cap.read()
H, W = frame_bgr.shape[:2]
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

prob = infer_mask2former_probs(m2f, processor, frame_rgb, H, W, device, use_fp16=True).cpu().numpy()
pred = np.argmax(prob, axis=0)
road = (pred == 0).astype(np.uint8) * 255

for y in [400, 450, 500, 550, 600, 650]:
    xs = np.where(road[y] > 0)[0]
    if len(xs) > 0:
        print(f"y={y}: min_road_x={xs[0]}, max_road_x={xs[-1]}")
    else:
        print(f"y={y}: no road")

