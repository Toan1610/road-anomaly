import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
print("Labels:", labels)

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx == 192:
        H, W = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Post-process semantic segmentation
            sem_seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0].cpu().numpy()
            
            # Sand pile box approx [0, 480, 250, 200]
            crop = sem_seg[480:680, 0:250]
            unq, cnts = np.unique(crop, return_counts=True)
            for u, c in zip(unq, cnts):
                print(f"  Class {u} ({labels.get(u, 'unknown')}): {c} pixels ({c/crop.size*100:.1f}%)")
        break

cap.release()
