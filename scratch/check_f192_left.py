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

# Check what is at y=550, x=0..400
y = 550
labels = model.config.id2label
print("Frame 192 at y=550:")
for x in range(0, 400, 20):
    c = pred[y, x]
    print(f"  x={x}: {c} ({labels.get(c, 'unknown')})")

