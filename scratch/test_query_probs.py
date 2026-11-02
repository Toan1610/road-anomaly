import cv2
import torch
import numpy as np
import torch.nn.functional as F
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

MEAN_GPU = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
STD_GPU = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
pixel_mask = torch.ones((1, 384, 384), device=device, dtype=torch.int64)

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx == 170:
        H, W = frame.shape[:2]
        t_frame = torch.from_numpy(frame).to(device, non_blocking=True)
        t_rgb = t_frame[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(torch.float16) / 255.0
        t_resized = F.interpolate(t_rgb, size=(384, 384), mode="bilinear", align_corners=False)
        t_norm = (t_resized - MEAN_GPU) / STD_GPU
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(pixel_values=t_norm, pixel_mask=pixel_mask)
            class_queries_logits = outputs.class_queries_logits # [1, 100, 20]
            masks_queries_logits = outputs.masks_queries_logits # [1, 100, 96, 96]
            
            # Check queries that have significant mask activation in the lower-left region
            # (y: 50-90, x: 0-30 in 96x96 grid)
            mask_sig = masks_queries_logits.sigmoid()[0] # [100, 96, 96]
            roi_sig = mask_sig[:, 55:85, 0:30].mean(dim=(1, 2)) # [100]
            top_q = torch.topk(roi_sig, 5)
            print("Top queries in ROI at frame 170:")
            for q_idx, score in zip(top_q.indices, top_q.values):
                q = q_idx.item()
                logits = class_queries_logits[0, q]
                probs = logits.softmax(dim=-1)
                top_cls = torch.topk(probs, 3)
                cls_desc = []
                for c_idx, p in zip(top_cls.indices, top_cls.values):
                    c = c_idx.item()
                    label = model.config.id2label.get(c, "null" if c==19 else str(c))
                    cls_desc.append(f"{label}: {p.item()*100:.1f}%")
                print(f"  Query {q:2d} (roi_activation={score.item():.2f}): {', '.join(cls_desc)}")
        break
cap.release()
