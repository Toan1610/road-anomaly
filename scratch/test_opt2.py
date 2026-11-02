import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

MEAN_GPU = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
STD_GPU = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

cap = cv2.VideoCapture("video/test2.mp4")
ret, frame_bgr = cap.read()
cap.release()
H, W = frame_bgr.shape[:2]

target_proc_h = 384
target_proc_w = 384
pixel_mask = torch.ones((1, target_proc_h, target_proc_w), device=device, dtype=torch.int64)

def infer_gpu_current(frame_bgr):
    t_frame = torch.from_numpy(frame_bgr).to(device, non_blocking=True)
    t_rgb = t_frame[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(torch.float16) / 255.0
    t_resized = F.interpolate(t_rgb, size=(target_proc_h, target_proc_w), mode="bilinear", align_corners=False)
    t_norm = (t_resized - MEAN_GPU) / STD_GPU
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=t_norm, pixel_mask=pixel_mask)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        masks_queries_logits = F.interpolate(
            masks_queries_logits, size=(H, W), mode="bilinear", align_corners=False
        )
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        probs = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)[0]
    return probs

def infer_gpu_lowres_einsum(frame_bgr):
    t_frame = torch.from_numpy(frame_bgr).to(device, non_blocking=True)
    t_rgb = t_frame[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(torch.float16) / 255.0
    t_resized = F.interpolate(t_rgb, size=(target_proc_h, target_proc_w), mode="bilinear", align_corners=False)
    t_norm = (t_resized - MEAN_GPU) / STD_GPU
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=t_norm, pixel_mask=pixel_mask)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits # [1, 100, 96, 96]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1] # [1, 100, 19]
        masks_probs = masks_queries_logits.sigmoid() # [1, 100, 96, 96]
        # Low-res einsum: [1, 19, 96, 96]
        probs_low = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        # Interpolate 19 channels to (H, W)
        probs = F.interpolate(probs_low, size=(H, W), mode="bilinear", align_corners=False)[0]
    return probs

# Warmup
for _ in range(5):
    p1 = infer_gpu_current(frame_bgr)
    p2 = infer_gpu_lowres_einsum(frame_bgr)
torch.cuda.synchronize()

# Benchmark Current
t0 = time.perf_counter()
for _ in range(40):
    p1 = infer_gpu_current(frame_bgr)
torch.cuda.synchronize()
t_cur = (time.perf_counter() - t0) / 40

# Benchmark Low-res einsum
t0 = time.perf_counter()
for _ in range(40):
    p2 = infer_gpu_lowres_einsum(frame_bgr)
torch.cuda.synchronize()
t_low = (time.perf_counter() - t0) / 40

print(f"Current GPU time: {t_cur*1000:.2f} ms ({1/t_cur:.1f} FPS)")
print(f"Low-res einsum GPU time: {t_low*1000:.2f} ms ({1/t_low:.1f} FPS) -> {t_cur/t_low:.2f}x speedup")

pred1 = p1.argmax(dim=0).cpu().numpy()
pred2 = p2.argmax(dim=0).cpu().numpy()
match = (pred1 == pred2).mean() * 100
print(f"Segmentation mask match: {match:.2f}%")
