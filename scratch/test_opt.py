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

# Pre-allocate ImageNet tensors on CUDA
MEAN_GPU = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
STD_GPU = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

cap = cv2.VideoCapture("video/test2.mp4")
ret, frame_bgr = cap.read()
cap.release()
H, W = frame_bgr.shape[:2]

# Method 1: CPU PIL + Processor
from PIL import Image
def infer_cpu(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        masks_queries_logits = F.interpolate(
            masks_queries_logits, size=(H, W), mode="bilinear", align_corners=False
        )
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        probs = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)[0]
    return probs

# Method 2: Pure GPU CUDA Preproc
# Check processor size settings
print("Processor size:", processor.size)
target_proc_h = processor.size.get("height", 384)
target_proc_w = processor.size.get("width", 384)
print(f"Target size: {target_proc_h}x{target_proc_w}")

def infer_gpu(frame_bgr):
    # frame_bgr is (H, W, 3) uint8 numpy
    t_frame = torch.from_numpy(frame_bgr).to(device, non_blocking=True) # (H, W, 3)
    # BGR to RGB: flip along dim 2, permute to (1, 3, H, W)
    t_rgb = t_frame[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(torch.float16) # (1, 3, H, W)
    # Scale 0-255 to 0-1
    t_rgb = t_rgb / 255.0
    # Resize to model input size (e.g. 384x384 or whatever processor uses)
    # Note: Hugging Face processor resizes smallest edge or fixed size.
    # Let's check how processor resized frame_rgb
    t_resized = F.interpolate(t_rgb, size=(target_proc_h, target_proc_w), mode="bilinear", align_corners=False)
    # Normalize with ImageNet mean and std
    t_norm = (t_resized - MEAN_GPU) / STD_GPU
    pixel_mask = torch.ones((1, target_proc_h, target_proc_w), device=device, dtype=torch.int64)
    
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

# Warmup
for _ in range(5):
    p1 = infer_cpu(frame_bgr)
    p2 = infer_gpu(frame_bgr)
torch.cuda.synchronize()

# Benchmark CPU
t0 = time.perf_counter()
for _ in range(30):
    p1 = infer_cpu(frame_bgr)
torch.cuda.synchronize()
t_cpu = (time.perf_counter() - t0) / 30

# Benchmark GPU
t0 = time.perf_counter()
for _ in range(30):
    p2 = infer_gpu(frame_bgr)
torch.cuda.synchronize()
t_gpu = (time.perf_counter() - t0) / 30

print(f"CPU Preproc time: {t_cpu*1000:.2f} ms")
print(f"GPU Preproc time: {t_gpu*1000:.2f} ms ({t_cpu/t_gpu:.2f}x speedup)")

pred1 = p1.argmax(dim=0).cpu().numpy()
pred2 = p2.argmax(dim=0).cpu().numpy()
match = (pred1 == pred2).mean() * 100
print(f"Segmentation mask match: {match:.2f}%")
