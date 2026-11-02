"""
Road Segmentation - Video Inference Script
Sử dụng YOLOv8s-seg với model best.pt để phân vùng đường từ video
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Road Segmentation Inference - Video Input",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python run_inference.py --input video.mp4
  python run_inference.py --input video.mp4 --output output_road.mp4
  python run_inference.py --input video.mp4 --conf 0.4 --device cpu
  python run_inference.py --input video.mp4 --no-show
  python run_inference.py --input 0   # webcam
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Đường dẫn video đầu vào (hoặc '0' để dùng webcam)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Road_Segmentation_Model/best.pt",
        help="Đường dẫn tới file model weights (mặc định: Road_Segmentation_Model/best.pt)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Đường dẫn video đầu ra (mặc định: tự động đặt tên theo input)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng confidence (mặc định: 0.25 - YOLO default)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Ngưỡng IoU cho NMS (mặc định: 0.7 - theo training config)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh inference (mặc định: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị: 'cpu', '0' (GPU 0), 'cuda' (mặc định: tự động chọn)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Không hiển thị cửa sổ preview khi chạy"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Chỉ detect class nhất định (0=object, 1=road). Mặc định: tất cả"
    )
    parser.add_argument(
        "--retina-masks",
        action="store_true",
        help="Dùng retina masks (chất lượng cao hơn nhưng chậm hơn)"
    )
    return parser.parse_args()


def resolve_output_path(input_path: str) -> str:
    """Tự động tạo tên file output dựa trên input."""
    p = Path(input_path)
    output_name = f"{p.stem}_road_seg{p.suffix}"
    return str(p.parent / output_name)


def main():
    args = parse_args()

    # --- Kiểm tra model ---
    model_path = args.model
    if not os.path.isfile(model_path):
        print(f"[LỖI] Không tìm thấy model: '{model_path}'")
        print("      Hãy kiểm tra lại đường dẫn với --model <path>")
        sys.exit(1)

    # --- Kiểm tra input ---
    input_src = args.input
    if input_src != "0" and not os.path.isfile(input_src):
        print(f"[LỖI] Không tìm thấy file video: '{input_src}'")
        sys.exit(1)

    # --- Đường dẫn output ---
    output_path = args.output
    if output_path is None and input_src != "0":
        output_path = resolve_output_path(input_src)

    # --- Load model ---
    print(f"\n{'='*55}")
    print(f"  Road Segmentation - Video Inference")
    print(f"{'='*55}")
    print(f"  Model   : {model_path}")
    print(f"  Input   : {input_src}")
    print(f"  Output  : {output_path or 'Khong luu (webcam)'}")
    print(f"  Conf    : {args.conf}  (YOLO default: 0.25)")
    print(f"  IoU     : {args.iou}  (training: 0.7)")
    print(f"  Imgsz   : {args.imgsz}")
    print(f"  Device  : {args.device or 'auto'}")
    print(f"  Classes : {args.classes or 'all (0=object, 1=road)'}")
    print(f"{'='*55}\n")

    print("[INFO] Dang load model...")
    model = YOLO(model_path)

    # --- Chạy inference ---
    print("[INFO] Bat dau inference...\n")

    predict_kwargs = dict(
        source=input_src,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=True,                   # lưu video output
        show=not args.no_show,       # hiển thị preview
        stream=True,                 # stream để không bị out of memory với video dài
        retina_masks=args.retina_masks,
        classes=args.classes,
        project=str(Path(output_path).parent) if output_path else "runs/segment",
        name=Path(output_path).stem if output_path else "predict",
        exist_ok=True,
    )

    if args.device is not None:
        predict_kwargs["device"] = args.device

    frame_count = 0
    try:
        for result in model.predict(**predict_kwargs):
            frame_count += 1
            # In progress mỗi 30 frame
            if frame_count % 30 == 0:
                boxes = len(result.boxes) if result.boxes is not None else 0
                print(f"  [Frame {frame_count:>5}] Detections: {boxes}")

    except KeyboardInterrupt:
        print("\n[INFO] Dung boi nguoi dung (Ctrl+C)")

    print(f"\n{'='*55}")
    print(f"  Hoan thanh! Da xu ly {frame_count} frames.")
    if output_path:
        saved_dir = Path(predict_kwargs["project"]) / predict_kwargs["name"]
        print(f"  Ket qua luu tai: {saved_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
