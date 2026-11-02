import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import transformers
transformers.logging.set_verbosity_error()

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

VIDEO_PATH = "video/test2.mp4"
OUTPUT_DIR = Path("/home/johnsmith/.gemini/antigravity-ide/brain/10be3a3e-3805-4b80-8851-e889c0de8c25/scratch/road_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Extract test frames from test2.mp4
TEST_FRAMES_IDX = [50, 100, 170, 192]
cap = cv2.VideoCapture(VIDEO_PATH)
frames = {}
cur_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if cur_idx in TEST_FRAMES_IDX:
        frames[cur_idx] = frame
    cur_idx += 1
    if cur_idx > max(TEST_FRAMES_IDX):
        break
cap.release()

print(f"[INFO] Extracted {len(frames)} frames: {list(frames.keys())}")
for f_idx, f_img in frames.items():
    cv2.imwrite(str(OUTPUT_DIR / f"frame_{f_idx}_orig.jpg"), f_img)

# Define inference functions
def infer_m2f_custom(model, frame_rgb, target_h, target_w, in_h=384, in_w=384):
    mean_gpu = torch.tensor([0.485, 0.456, 0.406], device=DEVICE, dtype=torch.float16).view(1, 3, 1, 1)
    std_gpu = torch.tensor([0.229, 0.224, 0.225], device=DEVICE, dtype=torch.float16).view(1, 3, 1, 1)
    pixel_mask = torch.ones((1, in_h, in_w), device=DEVICE, dtype=torch.int64)

    t_frame = torch.from_numpy(frame_rgb).to(DEVICE, non_blocking=True)
    t_chw = t_frame.permute(2, 0, 1).unsqueeze(0).to(torch.float16) / 255.0
    t_resized = F.interpolate(t_chw, size=(in_h, in_w), mode="bilinear", align_corners=False)
    t_norm = (t_resized - mean_gpu) / std_gpu

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=t_norm, pixel_mask=pixel_mask)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        probs_low = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        probs = F.interpolate(probs_low, size=(target_h, target_w), mode="bilinear", align_corners=False)[0]
    return probs

def infer_m2f_native_proc(model, processor, frame_rgb, target_h, target_w):
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        # Einsum at native pixel decoder resolution (e.g. 128x256 or 180x320)
        probs_low = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        probs = F.interpolate(probs_low, size=(target_h, target_w), mode="bilinear", align_corners=False)[0]
    return probs

def infer_segformer(model, processor, frame_rgb, target_h, target_w):
    image = Image.fromarray(frame_rgb)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**inputs)
        logits = F.interpolate(outputs.logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits[0], dim=0)
    return probs

# Models to test:
MODELS = {
    "m2f_tiny_384": {
        "type": "m2f_fast",
        "repo": "facebook/mask2former-swin-tiny-cityscapes-semantic",
        "res": (384, 384)
    },
    "m2f_tiny_native": {
        "type": "m2f_native",
        "repo": "facebook/mask2former-swin-tiny-cityscapes-semantic",
    },
    "m2f_large_384": {
        "type": "m2f_fast",
        "repo": "facebook/mask2former-swin-large-cityscapes-semantic",
        "res": (384, 384)
    },
    "m2f_large_native": {
        "type": "m2f_native",
        "repo": "facebook/mask2former-swin-large-cityscapes-semantic",
    },
    "segformer_b5": {
        "type": "segformer",
        "repo": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    },
}

results = {}

for name, cfg in MODELS.items():
    print(f"\n=======================================================")
    print(f"Testing model: {name} ({cfg['repo']})")
    print(f"=======================================================")
    t0 = time.time()
    if cfg["type"].startswith("m2f"):
        processor = Mask2FormerImageProcessor.from_pretrained(cfg["repo"])
        model = Mask2FormerForUniversalSegmentation.from_pretrained(cfg["repo"]).to(DEVICE)
        model.eval()
    else:
        processor = AutoImageProcessor.from_pretrained(cfg["repo"])
        model = SegformerForSemanticSegmentation.from_pretrained(cfg["repo"]).to(DEVICE)
        model.eval()

    # Param count
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    load_time = time.time() - t0
    print(f"[INFO] Loaded {name} in {load_time:.2f}s | Params: {num_params:.1f}M")

    # Warmup
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(2):
        if cfg["type"] == "m2f_fast":
            _ = infer_m2f_custom(model, dummy, 720, 1280, in_h=cfg["res"][0], in_w=cfg["res"][1])
        elif cfg["type"] == "m2f_native":
            _ = infer_m2f_native_proc(model, processor, dummy, 720, 1280)
        else:
            _ = infer_segformer(model, processor, dummy, 720, 1280)
    torch.cuda.synchronize()

    # Benchmark latency on frames
    latencies = []
    model_preds = {}

    for f_idx in TEST_FRAMES_IDX:
        frame_bgr = frames[f_idx]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        torch.cuda.synchronize()
        t_start = time.time()

        if cfg["type"] == "m2f_fast":
            probs = infer_m2f_custom(model, frame_rgb, h, w, in_h=cfg["res"][0], in_w=cfg["res"][1])
        elif cfg["type"] == "m2f_native":
            probs = infer_m2f_native_proc(model, processor, frame_rgb, h, w)
        else:
            probs = infer_segformer(model, processor, frame_rgb, h, w)

        torch.cuda.synchronize()
        latencies.append((time.time() - t_start) * 1000)

        # Extract predictions
        max_prob_t, pred_t = torch.max(probs, dim=0)
        pred_np = pred_t.cpu().numpy()
        road_prob_np = probs[0].cpu().numpy()  # Class 0: Road

        # Road mask (prob > 0.5)
        road_prob_np = probs[0].float().cpu().numpy()  # Class 0: Road (float32)
        road_mask = (pred_np == 0).astype(np.uint8) * 255
        road_prob_u8 = np.clip(road_prob_np * 255.0, 0, 255).astype(np.uint8)

        # Measure boundary sharpness: Laplacian of the road probability map along the edge
        # Find contours of road mask
        contours, _ = cv2.findContours((road_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(road_mask)
        cv2.drawContours(edge_mask, contours, -1, 255, 3)

        # Gradient magnitude along the edge
        sobelx = cv2.Sobel(road_prob_np, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(road_prob_np, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_sharpness = float(grad_mag[edge_mask > 0].mean()) if (edge_mask > 0).any() else 0.0

        model_preds[f_idx] = {
            "road_mask": road_mask,
            "road_prob": road_prob_u8,
            "pred": pred_np,
            "edge_sharpness": edge_sharpness,
            "road_area_pct": float((road_mask > 0).mean() * 100)
        }

        # Save visualization: Overlay road mask in green on frame
        vis = frame_bgr.copy()
        color_road = np.zeros_like(vis)
        color_road[:, :, 1] = 255 # Green
        vis = np.where(road_mask[:, :, None] > 0, cv2.addWeighted(vis, 0.6, color_road, 0.4, 0), vis)
        # Draw contour
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2) # Yellow border
        cv2.putText(vis, f"{name} (Edge Sharpness: {edge_sharpness:.3f})", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(str(OUTPUT_DIR / f"{name}_frame_{f_idx}_vis.jpg"), vis)
        cv2.imwrite(str(OUTPUT_DIR / f"{name}_frame_{f_idx}_mask.png"), road_mask)

    avg_lat = np.mean(latencies)
    avg_sharpness = np.mean([model_preds[f]["edge_sharpness"] for f in TEST_FRAMES_IDX])
    vram_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 / 1024

    results[name] = {
        "params_m": num_params,
        "avg_latency_ms": avg_lat,
        "fps": 1000.0 / avg_lat,
        "avg_sharpness": avg_sharpness,
        "vram_mb": vram_mb,
        "preds": model_preds
    }
    print(f"[RESULT] {name}: Avg Latency = {avg_lat:.2f} ms ({1000/avg_lat:.1f} FPS) | Sharpness = {avg_sharpness:.4f} | VRAM = {vram_mb:.0f} MB")

    # Clean up GPU
    del model, processor
    torch.cuda.empty_cache()

# Now compute Cross-Model Road Agreement (IoU) on all test frames
print("\n=======================================================")
print("Cross-Model Road Mask IoU Agreement Matrix (Frame 192 - Sand Encroachment):")
print("=======================================================")
f_eval = 192
model_names = list(MODELS.keys())
print(f"{'Model':<20}" + "".join([f"{m[:12]:>15}" for m in model_names]))
for m1 in model_names:
    row_str = f"{m1:<20}"
    mask1 = results[m1]["preds"][f_eval]["road_mask"] > 0
    for m2 in model_names:
        mask2 = results[m2]["preds"][f_eval]["road_mask"] > 0
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = (intersection / union) * 100 if union > 0 else 0
        row_str += f"{iou:>14.1f}%"
    print(row_str)

# Specifically inspect the Sand Pile region in Frame 192!
# Let's see what each model classifies in the sand pile region (bottom-right: y in [450, 650], x in [850, 1200])
print("\n=======================================================")
print("Frame 192 - Sand Pile Region Classification Analysis:")
print("ROI: y in [480, 620], x in [900, 1150]")
print("=======================================================")
for name in model_names:
    pred_roi = results[name]["preds"][192]["pred"][480:620, 900:1150]
    unique, counts = np.unique(pred_roi, return_counts=True)
    total = pred_roi.size
    class_dist = {int(u): float(c / total * 100) for u, c in zip(unique, counts)}
    # Top 3 classes in sand ROI
    sorted_cls = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    cls_names = {0: "road", 1: "sidewalk", 2: "building", 8: "vegetation", 9: "terrain", 18: "bicycle", -1: "uncertain"}
    cls_summary = ", ".join([f"{cls_names.get(c, str(c))}: {pct:.1f}%" for c, pct in sorted_cls])
    print(f"[{name:<18}] Top classes in Sand ROI: {cls_summary}")

# Save comparison montage for Frame 192
crops = []
for name in model_names:
    vis = cv2.imread(str(OUTPUT_DIR / f"{name}_frame_192_vis.jpg"))
    # Crop the right boundary / sand area: [350:700, 700:1280]
    crop = vis[350:700, 700:1280]
    cv2.putText(crop, name, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    crops.append(crop)

# Stack top 3 and bottom 2
top_row = np.hstack(crops[:3])
bottom_row = np.hstack(crops[3:] + [np.zeros_like(crops[0])])
montage = np.vstack([top_row, bottom_row])
cv2.imwrite(str(OUTPUT_DIR / "comparison_montage_frame_192.jpg"), montage)
print(f"\n[INFO] Saved comparison montage to: {OUTPUT_DIR / 'comparison_montage_frame_192.jpg'}")
