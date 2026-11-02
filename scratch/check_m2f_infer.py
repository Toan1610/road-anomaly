import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from run_video_encroachment_mask2former import infer_mask2former_probs

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
    if frame_idx == 192:
        H, W = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        probs = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, use_fp16=True)
        pred = torch.argmax(probs, dim=0).cpu().numpy()
        crop = pred[480:680, 0:250]
        unq, cnts = np.unique(crop, return_counts=True)
        print("infer_mask2former_probs prediction in sand pile:")
        for u, c in zip(unq, cnts):
            print(f"  Class {u} ({model.config.id2label.get(u, 'unknown')}): {c} pixels ({c/crop.size*100:.1f}%)")
        break
cap.release()
