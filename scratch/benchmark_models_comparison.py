import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation

# Setup path for PIDNet
PIDNET_ROOT = os.path.abspath("PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)
    sys.path.insert(0, os.path.join(PIDNET_ROOT, "tools"))

import models as pidnet_models

# Cityscapes Color map (BGR)
COLOR_MAP_BGR = np.array([
    (128,  64, 128),  # 0  road (tím / xám Cityscapes)
    (244,  35, 232),  # 1  sidewalk (hồng phấn)
    ( 70,  70,  70),  # 2  building
    (102, 102, 156),  # 3  wall
    (190, 153, 153),  # 4  fence
    (153, 153, 153),  # 5  pole
    (250, 170,  30),  # 6  traffic light
    (220, 220,   0),  # 7  traffic sign
    (107, 142,  35),  # 8  vegetation (xanh lá)
    (152, 251, 152),  # 9  terrain (xanh lá mạ)
    ( 70, 130, 180),  # 10 sky
    (220,  20,  60),  # 11 person
    (255,   0,   0),  # 12 rider
    (  0,   0, 142),  # 13 car (xanh dương đậm)
    (  0,   0,  70),  # 14 truck
    (  0,  60, 100),  # 15 bus
    (  0,  80, 100),  # 16 train
    (  0,   0, 230),  # 17 motorcycle
    (119,  11,  32),  # 18 bicycle
], dtype=np.uint8)

MEAN_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pidnet(arch, ckpt_path):
    model = pidnet_models.pidnet.get_pred_model(arch, 19)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v
        for k, v in ckpt.items()
        if k[6:] in model_dict and v.shape == model_dict[k[6:]].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    model = model.cuda().eval().half()
    return model


def load_segformer_b2():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    )
    model = model.cuda().eval().half()
    return model


def benchmark_model(name, infer_fn, sample_frame, n_warmup=15, n_runs=100):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / (1024 ** 2)

    # Warmup
    for _ in range(n_warmup):
        _ = infer_fn(sample_frame)
    torch.cuda.synchronize()

    # Benchmark
    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = infer_fn(sample_frame)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    mean_ms = np.mean(timings)
    std_ms = np.std(timings)
    min_ms = np.min(timings)
    fps = 1000.0 / mean_ms

    return {
        "name": name,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "fps": fps,
        "peak_mem_mb": peak_mem,
    }


def main():
    print("=" * 70)
    print("BENCHMARK CÁC MÔ HÌNH PHÂN ĐOẠN ĐƯỜNG & DỊ VẬT TRÊN MÁY HIỆN TẠI")
    print(f"Device: {torch.cuda.get_device_name(0)} | CUDA FP16")
    print("=" * 70)

    # 1. Đọc frame mẫu từ test2.mp4
    cap = cv2.VideoCapture("video/test2.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_f = 140  # Frame có đống cát rõ ràng
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
    ret, frame_bgr = cap.read()
    cap.release()
    assert ret, f"Không đọc được frame {target_f}"
    h, w = frame_bgr.shape[:2]
    print(f"Frame mẫu: {w}x{h} (Frame {target_f}/{total_frames})")

    # Danh sách models
    models_dict = {}

    # SegFormer B2
    print("\n[1/4] Đang nạp SegFormer B2...")
    seg_model = load_segformer_b2()

    def infer_segformer(img_bgr):
        # Resize về 1024x512 để cân bằng
        inp = cv2.resize(img_bgr, (1024, 512))
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN_IMAGENET) / STD_IMAGENET
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).cuda().half()
        with torch.no_grad():
            outputs = seg_model(pixel_values=tensor)
            logits = F.interpolate(outputs.logits.float(), size=(h, w), mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        return pred

    models_dict["SegFormer-B2 (1024x512)"] = infer_segformer

    # PIDNet-S
    print("[2/4] Đang nạp PIDNet-S...")
    pidnet_s = load_pidnet("pidnet-s", "PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt")

    def infer_pidnet_s(img_bgr):
        inp = cv2.resize(img_bgr, (1024, 512))
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN_IMAGENET) / STD_IMAGENET
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).cuda().half()
        with torch.no_grad():
            out = pidnet_s(tensor)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            logits = F.interpolate(out.float(), size=(h, w), mode="bilinear", align_corners=True)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        return pred

    models_dict["PIDNet-S (1024x512)"] = infer_pidnet_s

    # PIDNet-M
    print("[3/4] Đang nạp PIDNet-M...")
    pidnet_m = load_pidnet("pidnet-m", "PIDNet/pretrained_models/cityscapes/PIDNet_M_Cityscapes_val.pt")

    def infer_pidnet_m(img_bgr):
        inp = cv2.resize(img_bgr, (1024, 512))
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN_IMAGENET) / STD_IMAGENET
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).cuda().half()
        with torch.no_grad():
            out = pidnet_m(tensor)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            logits = F.interpolate(out.float(), size=(h, w), mode="bilinear", align_corners=True)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        return pred

    models_dict["PIDNet-M (1024x512)"] = infer_pidnet_m

    # YOLO26n-sem
    print("[4/4] Đang nạp YOLO26n-sem...")
    yolo_sem = YOLO("yolo26n-sem.pt")

    def infer_yolo(img_bgr):
        results = yolo_sem.predict(
            img_bgr,
            imgsz=640,
            conf=0.25,
            device=0,
            half=True,
            verbose=False,
        )
        r = results[0]
        if r.masks is not None and len(r.masks) > 0:
            pred = np.full((h, w), 255, dtype=np.uint8)
            classes = r.boxes.cls.cpu().numpy().astype(int)
            masks = r.masks.data.cpu().numpy()
            for cls_id, m in zip(classes, masks):
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred[m > 0.5] = cls_id
        else:
            pred = np.full((h, w), 255, dtype=np.uint8)
        return pred

    models_dict["YOLO26n-sem (640x640)"] = infer_yolo

    # Run Benchmark
    print("\n" + "=" * 70)
    print("BẮT ĐẦU ĐO ĐỘ TRỄ & TỐC ĐỘ (100 ITERATIONS PER MODEL)...")
    print("=" * 70)

    results = []
    preds = {}
    for name, fn in models_dict.items():
        print(f"Đang benchmark {name}...")
        res = benchmark_model(name, fn, frame_bgr, n_warmup=15, n_runs=100)
        results.append(res)
        # Lưu lại mask để trực quan hóa
        preds[name] = fn(frame_bgr)
        print(f"   -> Latency: {res['mean_ms']:.2f} ms ± {res['std_ms']:.2f} ms | FPS: {res['fps']:.1f} | VRAM: {res['peak_mem_mb']:.1f} MB")

    # In bảng tổng hợp
    print("\n" + "=" * 75)
    print(f"{'MÔ HÌNH':<28} | {'LATENCY (ms)':<14} | {'FPS THỰC TẾ':<12} | {'VRAM (MB)':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<28} | {r['mean_ms']:>6.2f} ± {r['std_ms']:>4.2f} ms | {r['fps']:>10.1f} FPS | {r['peak_mem_mb']:>8.1f} MB")
    print("=" * 75)

    # 4. Tạo ảnh so sánh trực quan
    os.makedirs("scratch/benchmark_output", exist_ok=True)

    # Trực quan hóa từng model
    vis_panels = []
    for name, pred in preds.items():
        # Overlay Cityscapes classes
        vis = frame_bgr.copy()
        # Tạo color mask
        valid = (pred >= 0) & (pred < 19)
        color_layer = np.zeros_like(frame_bgr)
        color_layer[valid] = COLOR_MAP_BGR[pred[valid]]

        # Road (tím), Sidewalk (hồng), Obstacle (vàng/xanh khác)
        # Trộn màu 50%
        overlay = cv2.addWeighted(vis, 0.55, color_layer, 0.45, 0)

        # Highlight riêng vùng ROAD (xanh lá cây) và VẬT THỂ KHÔNG PHẢI ROAD/SIDEWALK trên đường
        road_mask = (pred == 0).astype(np.uint8)
        sw_mask = (pred == 1).astype(np.uint8)

        # Ghi text thông tin
        fps_val = next(r["fps"] for r in results if r["name"] == name)
        ms_val = next(r["mean_ms"] for r in results if r["name"] == name)
        vram_val = next(r["peak_mem_mb"] for r in results if r["name"] == name)

        cv2.rectangle(overlay, (10, 10), (450, 95), (0, 0, 0), -1)
        cv2.putText(overlay, f"{name}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(overlay, f"Speed: {ms_val:.1f} ms ({fps_val:.1f} FPS) | VRAM: {vram_val:.0f} MB", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Thống kê pixel road
        road_px = int(np.sum(road_mask))
        sw_px = int(np.sum(sw_mask))
        cv2.putText(overlay, f"Road px: {road_px:,} | Sidewalk px: {sw_px:,}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        vis_panels.append(overlay)

    # Ghép 4 panel lại thành 2x2 grid
    top_row = np.hstack([vis_panels[0], vis_panels[1]])
    bot_row = np.hstack([vis_panels[2], vis_panels[3]])
    grid = np.vstack([top_row, bot_row])

    # Resize để vừa xem
    grid_resized = cv2.resize(grid, (1920, 1080))
    save_path = "scratch/benchmark_output/comparison_grid_f140.jpg"
    cv2.imwrite(save_path, grid_resized)
    print(f"\n[XONG] Đã lưu ảnh so sánh trực quan 4 model tại: {save_path}")

    # Tạo thêm ảnh crop cận cảnh đống cát để xem chi tiết
    # Đống cát nằm ở mép trái, tọa độ x: 100-550, y: 380-680
    crop_panels = []
    y1, y2, x1, x2 = 380, 680, 100, 550
    for name, pred in preds.items():
        sub_frame = frame_bgr[y1:y2, x1:x2].copy()
        sub_pred = pred[y1:y2, x1:x2]
        valid = (sub_pred >= 0) & (sub_pred < 19)
        color_layer = np.zeros_like(sub_frame)
        color_layer[valid] = COLOR_MAP_BGR[sub_pred[valid]]
        sub_overlay = cv2.addWeighted(sub_frame, 0.5, color_layer, 0.5, 0)
        cv2.rectangle(sub_overlay, (5, 5), (280, 40), (0, 0, 0), -1)
        cv2.putText(sub_overlay, name.split()[0], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        crop_panels.append(sub_overlay)

    crop_grid = np.hstack(crop_panels)
    crop_path = "scratch/benchmark_output/sand_pile_detail_crop.jpg"
    cv2.imwrite(crop_path, crop_grid)
    print(f"[XONG] Đã lưu ảnh crop cận cảnh đống cát tại: {crop_path}")


if __name__ == "__main__":
    main()
