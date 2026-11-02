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
    if frame_idx in [155, 160, 170, 180, 190]:
        H, W = frame.shape[:2]
        t_frame = torch.from_numpy(frame).to(device, non_blocking=True)
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
            
            # Check max probability in the sand region (y: 480-650, x: 50-250)
            crop_probs = probs[:, 480:650, 50:250]
            max_p, pred = torch.max(crop_probs, dim=0)
            pred_cls = pred.cpu().numpy()
            max_p_np = max_p.cpu().numpy()
            
            print(f"Frame {frame_idx:03d} in Sand ROI:")
            print(f"  Mean max_prob: {max_p_np.mean():.4f}, Min: {max_p_np.min():.4f}, Max: {max_p_np.max():.4f}")
            unq, cnts = np.unique(pred_cls, return_counts=True)
            for u, c in zip(unq, cnts):
                sub_p = max_p_np[pred_cls == u]
                print(f"    Class {u} ({model.config.id2label.get(u, u)}): {c} px, avg_conf = {sub_p.mean():.3f}")

cap.release()
