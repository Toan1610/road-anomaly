import subprocess
import time
import sys
import os
import re
import json
from pathlib import Path

MODELS_TO_TEST = [
    {
        "name": "PIDNet-S (Hybrid - YOLO + PIDNet-S)",
        "model_key": "pidnet-s",
        "mode": "hybrid",
        "desc": "Mô hình đề xuất chính (YOLO kẻ biên + PIDNet-S tách dị vật)",
    },
    {
        "name": "PIDNet-S (Standalone - Thuần PIDNet-S)",
        "model_key": "pidnet-s",
        "mode": "pidnet",
        "desc": "Mô hình PIDNet-S đơn độc lập",
    },
    {
        "name": "PIDNet-M (Hybrid - YOLO + PIDNet-M)",
        "model_key": "pidnet-m",
        "mode": "hybrid",
        "desc": "Mô hình PIDNet-M cỡ vừa",
    },
    {
        "name": "SegFormer-B2 (Hybrid - YOLO + SegFormer-B2)",
        "model_key": "b2",
        "mode": "hybrid",
        "desc": "Mô hình Transformer chuẩn của NVIDIA",
    },
    {
        "name": "YOLO26n-sem (Standalone - Thuần YOLO26n)",
        "model_key": "yolo26n",
        "mode": "yolo",
        "desc": "Mô hình siêu nhẹ đơn độc lập của Ultralytics",
    },
]

def run_test_for_model(config, video_path="video/test2.mp4", loops=3, max_frames=0):
    model_name = config["name"]
    model_key = config["model_key"]
    mode = config["mode"]
    out_dir = Path(f"output_endurance_{model_key}_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "./venv/bin/python",
        "run_video_encroachment.py",
        "--input", video_path,
        "--model", model_key,
        "--mode", mode,
        "--loop", str(loops),
        "--no-save_video",
        "--no-save_masks",
        "--output", str(out_dir),
    ]
    if max_frames > 0:
        cmd.extend(["--max_frames", str(max_frames)])

    print(f"\n{'='*75}")
    print(f" ĐANG KIỂM THỬ ĐỘ BỀN & RÒ RỈ BỘ NHỚ (ENDURANCE TEST): {model_name}")
    print(f" Lệnh: {' '.join(cmd)}")
    print(f"{'='*75}")

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    samples = []
    total_frames_processed = 0
    final_fps = 0.0

    # Pattern: [INFO] Frame 30/243 (12.3%) [Loop 1/3] | Đã xử lý: 30 frames | Tốc độ: 7.47 FPS | RAM: 2316.4 MB | VRAM: 131.9/494.0 MB
    pattern = re.compile(
        r"Đã xử lý:\s*(\d+)\s*frames\s*\|\s*Tốc độ:\s*([\d\.]+)\s*FPS\s*\|\s*RAM:\s*([\d\.]+)\s*MB(?:\s*\|\s*VRAM:\s*([\d\.]+)/([\d\.]+)\s*MB)?"
    )

    for line in proc.stdout:
        line_str = line.strip()
        print(line_str)
        m = pattern.search(line_str)
        if m:
            frames = int(m.group(1))
            fps = float(m.group(2))
            ram_mb = float(m.group(3))
            vram_alloc = float(m.group(4)) if m.group(4) else 0.0
            vram_res = float(m.group(5)) if m.group(5) else 0.0
            samples.append({
                "frame": frames,
                "fps": fps,
                "ram_mb": ram_mb,
                "vram_alloc": vram_alloc,
                "vram_res": vram_res,
            })
            total_frames_processed = frames
            final_fps = fps

    proc.wait()
    t_elapsed = time.perf_counter() - t0

    # Cleanup output folder
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)

    if proc.returncode != 0:
        print(f"[LỖI] Tiến trình kết thúc bất thường với mã lỗi {proc.returncode} (VĂNG APP)")
        return {
            "name": model_name,
            "success": False,
            "error_code": proc.returncode,
            "elapsed_s": t_elapsed,
        }

    # Phân tích độ ổn định bộ nhớ
    first_sample = samples[0] if samples else {"ram_mb": 0, "vram_alloc": 0, "vram_res": 0, "fps": 0}
    # Lấy sample sau warmup (ví dụ từ frame 90 trở đi)
    warmup_idx = min(2, len(samples) - 1) if len(samples) > 2 else 0
    post_warmup_sample = samples[warmup_idx] if samples else first_sample
    last_sample = samples[-1] if samples else first_sample

    ram_start = first_sample["ram_mb"]
    ram_stable = post_warmup_sample["ram_mb"]
    ram_end = last_sample["ram_mb"]
    vram_alloc_end = last_sample["vram_alloc"]
    vram_res_end = last_sample["vram_res"]

    ram_drift = ram_end - ram_stable
    frames_delta = max(1, last_sample["frame"] - post_warmup_sample["frame"])
    leak_rate_per_1000 = (ram_drift / frames_delta) * 1000

    # Dự đoán sau 24h chạy liên tục ở 25 FPS (2,160,000 frames)
    projected_24h_ram_growth_mb = leak_rate_per_1000 * (2160000 / 1000)

    res = {
        "name": model_name,
        "success": True,
        "total_frames": total_frames_processed,
        "elapsed_s": round(t_elapsed, 2),
        "fps": round(final_fps, 2),
        "ram_start_mb": round(ram_start, 1),
        "ram_stable_mb": round(ram_stable, 1),
        "ram_end_mb": round(ram_end, 1),
        "ram_drift_mb": round(ram_drift, 2),
        "leak_rate_per_1000_frames_mb": round(leak_rate_per_1000, 3),
        "vram_alloc_mb": round(vram_alloc_end, 1),
        "vram_res_mb": round(vram_res_end, 1),
        "projected_24h_growth_mb": round(projected_24h_ram_growth_mb, 1),
        "samples": samples,
    }
    return res

def main():
    print("BẮT ĐẦU CHƯƠNG TRÌNH KIỂM THỬ ĐỘ BỀN VẬN HÀNH 24/7 TẤT CẢ CÁC MÔ HÌNH...")
    all_results = []
    for cfg in MODELS_TO_TEST:
        res = run_test_for_model(cfg, video_path="video/test2.mp4", loops=3)
        all_results.append(res)
        time.sleep(2)  # nghỉ giữa các model để GPU giải phóng hoàn toàn

    print("\n" + "="*80)
    print(" KẾT QUẢ TỔNG HỢP KIỂM THỬ ĐỘ BỀN & ỔN ĐỊNH BỘ NHỚ 24/7")
    print("="*80)
    print(f"{'Mô hình':<40} | {'FPS':<6} | {'RAM Đầu':<8} | {'RAM Cuối':<8} | {'ΔRAM':<8} | {'VRAM Alloc':<10} | {'Nguy cơ Văng 24h':<16}")
    print("-" * 105)
    for r in all_results:
        if not r["success"]:
            print(f"{r['name']:<40} | LỖI VĂNG (Code {r['error_code']})")
            continue
        drift = r['ram_drift_mb']
        drift_str = f"{drift:+.1f} MB"
        risk = "AN TOÀN TUYỆT ĐỐI" if abs(drift) < 20 else ("CẦN THEO DÕI" if abs(drift) < 80 else "NGUY HIỂM TRÀN RAM")
        print(f"{r['name']:<40} | {r['fps']:<6.1f} | {r['ram_start_mb']:<6.1f}MB | {r['ram_end_mb']:<6.1f}MB | {drift_str:<8} | {r['vram_alloc_mb']:<8.1f}MB | {risk:<16}")

    out_json = Path("scratch/endurance_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[XONG] Đã lưu kết quả chi tiết vào: {out_json}")

if __name__ == "__main__":
    main()
