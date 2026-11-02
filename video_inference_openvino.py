import cv2
import time
import numpy as np
import openvino as ov
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================
ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "models/road-segmentation-adas-0001/"
    / "road-segmentation-adas-0001.xml"
)

VIDEO_PATH = ROOT / "video/test.mp4"
OUTPUT_PATH = ROOT / "outputs/test_openvino.mp4"

# CPU: chạy thử trên CPU trước
DEVICE = "CPU"

# Hệ số hòa trộn mask với video gốc
ALPHA = 0.35

# Model yêu cầu đầu vào 896x512
INPUT_WIDTH = 896
INPUT_HEIGHT = 512

# Màu BGR cho từng lớp
# 0 = BG, 1 = road, 2 = curb, 3 = mark
COLORS = np.array(
    [
        [0, 0, 0],       # BG: đen
        [0, 255, 0],     # Road: xanh lá
        [0, 0, 255],     # Curb: đỏ
        [255, 0, 0],     # Mark: xanh dương
    ],
    dtype=np.uint8,
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {VIDEO_PATH}")

    # =========================
    # NẠP MODEL
    # =========================
    core = ov.Core()
    model = core.read_model(str(MODEL_PATH))
    compiled_model = core.compile_model(model, DEVICE)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    print("Device:", DEVICE)
    print("Model input:", input_layer.shape)
    print("Model output:", output_layer.shape)

    # =========================
    # MỞ VIDEO
    # =========================
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {VIDEO_PATH}")

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps_input <= 0:
        fps_input = 25.0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Giữ nguyên độ phân giải và FPS video gốc
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps_input,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không thể tạo video: {OUTPUT_PATH}")

    print(f"Video: {width}x{height} @ {fps_input:.2f} FPS")
    print(f"Tổng số frame dự kiến: {total_frames}")
    print("Đang xử lý video...")

    frame_count = 0
    inference_time_sum = 0.0
    processing_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # =========================
            # TIỀN XỬ LÝ
            # =========================
            resized = cv2.resize(
                frame,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            # OpenVINO model nhận BGR, NCHW
            input_tensor = np.ascontiguousarray(
                resized.transpose(2, 0, 1)[None, ...]
            )

            # =========================
            # SUY LUẬN
            # =========================
            inference_start = time.perf_counter()

            result = compiled_model({input_layer: input_tensor})[output_layer]

            inference_time = time.perf_counter() - inference_start
            inference_time_sum += inference_time

            # [1, 4, 512, 896] -> [512, 896]
            mask = np.argmax(result[0], axis=0).astype(np.uint8)

            # =========================
            # TẠO MASK VÀ OVERLAY
            # =========================
            mask_color = COLORS[mask]

            # Phóng mask về độ phân giải video gốc
            mask_color = cv2.resize(
                mask_color,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Chỉ phủ màu lên vùng road, curb, mark
            valid = mask_color.any(axis=2)

            output_frame = frame.copy()

            blended = cv2.addWeighted(
                frame,
                1.0 - ALPHA,
                mask_color,
                ALPHA,
                0,
            )

            output_frame[valid] = blended[valid]

            # Hiển thị thông tin FPS trên video
            elapsed = time.perf_counter() - processing_start
            fps_current = (frame_count + 1) / elapsed if elapsed > 0 else 0

            cv2.putText(
                output_frame,
                f"OpenVINO {DEVICE} | FPS: {fps_current:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(output_frame)
            frame_count += 1

            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - processing_start
                fps_total = frame_count / elapsed if elapsed > 0 else 0

                print(
                    f"Frames: {frame_count}/{total_frames} | "
                    f"FPS tổng: {fps_total:.2f}"
                )

    finally:
        cap.release()
        writer.release()

    # =========================
    # TỔNG KẾT
    # =========================
    total_time = time.perf_counter() - processing_start

    fps_total = frame_count / total_time if total_time > 0 else 0
    fps_inference = (
        frame_count / inference_time_sum
        if inference_time_sum > 0
        else 0
    )

    print("\n========== KẾT QUẢ ==========")
    print(f"Frames đã xử lý: {frame_count}")
    print(f"Thời gian tổng: {total_time:.2f} giây")
    print(f"FPS tổng pipeline: {fps_total:.2f}")
    print(f"FPS suy luận riêng: {fps_inference:.2f}")
    print(f"Video kết quả: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
