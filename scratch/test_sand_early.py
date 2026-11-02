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

CITYSCAPES_19_LABELS = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole",
    6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain",
    10: "sky", 11: "person", 12: "rider", 13: "car", 14: "truck",
    15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
}

def infer_gpu_fast(frame_bgr, H, W):
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
    return probs, outputs

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
frames_to_check = [150, 155, 160, 165, 170, 175, 180, 185, 190]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx in frames_to_check:
        H, W = frame.shape[:2]
        probs, outputs = infer_gpu_fast(frame, H, W)
        pred = probs.argmax(dim=0).cpu().numpy()
        
        # Region where sand is on the left
        # Y from 450 to 650, X from 0 to 350
        crop = pred[450:650, 0:350]
        unq, cnts = np.unique(crop, return_counts=True)
        classes_str = ", ".join([f"{CITYSCAPES_19_LABELS.get(u, u)}: {c}" for u, c in zip(unq, cnts)])
        print(f"Frame {frame_idx:03d} - Sand crop classes: {classes_str}")

cap.release()
