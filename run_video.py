import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Danh sách các biến thể model SegFormer & Ultralytics YOLO26 Semantic
MODEL_CHOICES = {
    # SegFormer Cityscapes (HuggingFace Transformers)
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    "b4": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    # Ultralytics YOLO26 Semantic Segmentation (Cityscapes Pretrained)
    "yolo26n": "yolo26n-sem.pt",
    "yolo26s": "yolo26s-sem.pt",
    "yolo26m": "yolo26m-sem.pt",
    "yolo26l": "yolo26l-sem.pt",
    "yolo26x": "yolo26x-sem.pt",
}
DEFAULT_MODEL = "b2"

# Các class thuộc hành lang giao thông (drivable corridor)
# Bao gồm mặt đường và toàn bộ phương tiện/người di chuyển trên đường
CORRIDOR_CLASSES = {
    "road",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "rider",
    "person",
    "traffic sign",
    "traffic light",
}

# Bảng màu hiển thị (RGB)
COLORS = {
    "road": (0, 255, 0),          # Xanh lá: mặt đường xe chạy
    "sidewalk": (255, 165, 0),    # Cam: vỉa hè / lề đường
    "car": (255, 0, 0),           # Đỏ: ô tô con
    "motorcycle": (255, 255, 0),  # Vàng: xe máy
    "bicycle": (0, 255, 255),     # Cyan: xe đạp
    "bus": (255, 0, 255),         # Tím: xe buýt
    "truck": (255, 128, 0),       # Cam đỏ: xe tải
    "person": (255, 0, 128),      # Hồng: người đi bộ
    "rider": (255, 69, 0),        # Đỏ cam: người lái xe máy / xe đạp
}

PALETTE = {
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "white": (255, 255, 255),
    "lime": (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SegFormer Road Segmentation & Precise Encroachment Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "--video",
        type=str,
        default="/home/johnsmith/test/road_segmentation/video/test.mp4",
        help="Đường dẫn file video đầu vào",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Thư mục lưu các video kết quả",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Tên checkpoint hoặc alias (b2: cân bằng 10-12 FPS, b5: chính xác nhất)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "segformer", "yolo"],
        help="Chế độ xử lý: hybrid (YOLO kẻ biên chuẩn + SegFormer tách dị vật), segformer (chỉ SegFormer), yolo (chỉ YOLO)",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolo26n-sem.pt",
        help="Checkpoint YOLO Semantic dùng để kẻ đường biên chuẩn trong chế độ hybrid",
    )
    parser.add_argument(
        "--spatial_smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật bộ lọc không gian (Spatial Smoothing) làm mịn ma trận xác suất của SegFormer, khử răng cưa mép xanh lá",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Tỉ lệ co/giãn độ phân giải video đầu vào (0 < scale <= 1.0)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Bước nhảy frame: 1 (xử lý mọi frame), 2 (xử lý mỗi 2 frame, tăng gấp đôi tốc độ)",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tự động bật FP16 mixed precision khi có GPU CUDA",
    )

    # Ổn định thời gian
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.7,
        help="Hệ số làm mượt xác suất theo thời gian EMA (0.0: tắt, 0.7: mượt)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.35,
        help="Ngưỡng xác suất tin cậy tối thiểu cho từng pixel",
    )

    # Căn chỉnh đường thẳng mép đường chuẩn (Curb Boundary Lines)
    # Phương trình đường thẳng: x = a * y + b
    # Khớp chuẩn xác theo 2 đường biên sát biên đường do người dùng chỉ định
    parser.add_argument(
        "--left_line",
        type=str,
        default=None,
        help="Hệ số (a,b) cho đường thẳng mép biên TRÁI (x = a*y + b). Mặc định: None (Tự động thích ứng đa video & độ phân giải)",
    )
    parser.add_argument(
        "--right_line",
        type=str,
        default=None,
        help="Hệ số (a,b) cho đường thẳng mép biên PHẢI (x = a*y + b). Mặc định: None (Tự động thích ứng đa video & độ phân giải)",
    )
    parser.add_argument(
        "--boundary_color",
        type=str,
        default="yellow",
        choices=["yellow", "cyan", "white", "lime"],
        help="Màu sắc đường thẳng biên chuẩn (mép đường)",
    )
    parser.add_argument(
        "--min_encroach_area",
        type=int,
        default=1200,
        help="Diện tích tối thiểu (pixels) để xác định là vật thể tĩnh lấn chiếm (mặc định: 1200 px)",
    )
    parser.add_argument(
        "--min_encroach_depth",
        type=float,
        default=50.0,
        help="Độ sâu lấn chiếm tối thiểu (pixels) để tính là vi phạm (mặc định: 50 px, loại bỏ gờ chân cột/bóng xe mà vẫn bắt trọn đống cát lấn đường từ xa)",
    )
    parser.add_argument(
        "--static_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Chỉ nhận diện vật thể tĩnh cố định trên đường, tự động loại trừ vệt nhiễu/bóng do xe máy/ô tô di chuyển qua",
    )
    parser.add_argument(
        "--vehicle_buffer",
        type=int,
        default=35,
        help="Khoảng cách đệm (pixels) quanh phương tiện giao thông để loại bỏ bóng xe và vật thể di chuyển",
    )
    parser.add_argument(
        "--dynamic_lines",
        "--dynamic-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tự động nhận diện và khớp 2 đường biên mép đường real-time theo SegFormer trên từng frame",
    )
    parser.add_argument(
        "--line_smooth",
        "--line-smooth",
        type=float,
        default=0.15,
        help="Hệ số làm mượt cập nhật đường biên theo thời gian EMA (0.1 - 0.4)",
    )
    parser.add_argument(
        "--show_boundary_lines",
        "--show-boundary-lines",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hiển thị 2 đường kẻ biên mép đường màu vàng trên video overlay (mặc định: Ẩn)",
    )

    # Xuất mask video
    parser.add_argument(
        "--save_masks",
        "--save-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lưu thêm video road_mask.mp4 và encroach_mask.mp4 (road + xe = trắng, đống cát = đen)",
    )

    # Lưu bằng chứng vi phạm phục vụ hậu kiểm / gửi trung tâm
    parser.add_argument(
        "--save_incidents",
        "--save-incidents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tự động chụp và lưu ảnh bằng chứng các vụ lấn chiếm ra output/incidents/ cùng file log JSON",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.8,
        help="Thời gian theo dõi (tracking) tối thiểu tính bằng GIÂY để xác nhận là lấn chiếm thực sự (mặc định: 0.8s, vừa khử sạch nháy vừa cảnh báo sớm từ xa)",
    )
    parser.add_argument(
        "--incident_min_frames",
        type=int,
        default=8,
        help="Số frame liên tiếp phát hiện lấn chiếm tối thiểu để ghi nhận sự việc (tránh báo động giả)",
    )
    parser.add_argument(
        "--min_alarm_frames",
        type=int,
        default=4,
        help="Số frame liên tiếp phát hiện lấn chiếm tối thiểu để kích hoạt cảnh báo đỏ trên overlay (khử nháy góc)",
    )

    # Phát hiện hàng rong / buôn bán lấn chiếm lòng đường
    parser.add_argument(
        "--vendor_detect",
        "--vendor-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật phát hiện người bán hàng rong đứng yên gần mép đường (person + xe đạp/xe đẩy)",
    )
    parser.add_argument(
        "--vendor_min_sec",
        "--vendor-min-sec",
        type=float,
        default=1.5,
        help="Thời gian tối thiểu đứng yên (giây) để xác nhận là hàng rong lấn chiếm (mặc định: 1.5s)",
    )
    parser.add_argument(
        "--vendor_edge_thresh",
        "--vendor-edge-thresh",
        type=float,
        default=120.0,
        help="Khoảng cách tối đa (pixels) từ mép đường để xác định hàng rong sát biên lòng đường (mặc định: 120px)",
    )
    parser.add_argument(
        "--max_frames",
        "--max-frames",
        type=int,
        default=0,
        help="Giới hạn số frame xử lý (0 = xử lý toàn bộ video, hữu ích khi test nhanh)",
    )

    parser.add_argument(
        "--infra_thresh",
        "--infra-thresh",
        type=float,
        default=0.55,
        help="Ngưỡng tỷ lệ hạ tầng cố định (vỉa hè, bóng trụ cầu, cột đèn) theo YOLO để loại trừ báo động giả (mặc định: 0.55)",
    )

    # Độ trong suốt overlay
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Độ phủ màu lớp segmentation (0.0 đến 1.0)",
    )

    return parser.parse_args()


def parse_line_params(line_str, default_a, default_b):
    try:
        parts = [float(x.strip()) for x in line_str.split(",")]
        if len(parts) == 2:
            return parts[0], parts[1]
    except Exception:
        pass
    return default_a, default_b


def compute_line_endpoints(la, lb, ra, rb, y_top, y_bottom, width):
    """
    Tính tọa độ cắt thực tế của 2 đường biên với các mép màn hình:
    - Đường trái thoát khỏi màn hình tại mép trái x=0 khi y = -lb / la.
    - Đường phải thoát khỏi màn hình tại mép phải x=width-1 khi y = (width-1 - rb) / ra.
    Tránh ép gượng y_bottom làm xoay lệch đường kẻ vào lòng đường.
    """
    if abs(la) > 1e-4:
        y_exit_l = -lb / la
        y_l_bot = int(np.clip(y_exit_l, y_top, y_bottom))
    else:
        y_l_bot = y_bottom
    p_lt = (int(np.clip(la * y_top + lb, 0, width - 1)), y_top)
    p_lb = (int(np.clip(la * y_l_bot + lb, 0, width - 1)), y_l_bot)

    if abs(ra) > 1e-4:
        y_exit_r = (width - 1 - rb) / ra
        y_r_bot = int(np.clip(y_exit_r, y_top, y_bottom))
    else:
        y_r_bot = y_bottom
    p_rt = (int(np.clip(ra * y_top + rb, 0, width - 1)), y_top)
    p_rb = (int(np.clip(ra * y_r_bot + rb, 0, width - 1)), y_r_bot)

    return (p_lt, p_lb, y_l_bot), (p_rt, p_rb, y_r_bot)


def compute_corridor_polygon(la, lb, ra, rb, y_top, y_bottom, width, height):
    """
    Xây dựng mặt nạ đa giác hành lang lòng đường chuẩn khép kín theo đúng 2 đường biên.
    Tuyệt đối không tự ý nới góc đáy ra mép màn hình (width-1) gây tràn lên vỉa hè.
    """
    (p_lt, p_lb, y_l_bot), (p_rt, p_rb, y_r_bot) = compute_line_endpoints(
        la, lb, ra, rb, y_top, y_bottom, width
    )
    poly = []
    # 1. Đi dọc mép trái từ đỉnh y_top xuống đáy
    for y in range(y_top, y_l_bot + 1, 2):
        xl = int(la * y + lb)
        poly.append([max(0, min(width - 1, xl)), y])

    if y_l_bot < y_bottom:
        poly.append([0, y_bottom])

    # 2. Nối đáy sang mép phải
    if y_r_bot < y_bottom:
        poly.append([width - 1, y_bottom])
        poly.append([width - 1, y_r_bot])
    else:
        poly.append([p_rb[0], y_bottom])

    # 3. Đi ngược lên dọc theo mép phải từ đáy lên y_top
    for y in range(y_r_bot, y_top - 1, -2):
        xr = int(ra * y + rb)
        poly.append([max(0, min(width - 1, xr)), y])

    poly_mask = np.zeros((height, width), dtype=np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(poly_mask, [np.array(poly, dtype=np.int32)], 255)
    return poly_mask, (p_lt, p_lb), (p_rt, p_rb)


def is_corner_artifact(cnt, width, height, y_top, y_bottom):
    """
    Loại bỏ các dị vật giả tạo do hiện tượng mép góc màn hình / nắp capo (FOV Edge Clipping).
    Một vật thể lấn chiếm thực sự trên lòng đường phải vươn từ xa vào trong làn xe chạy (y < y_threshold),
    thay vì chỉ nằm ép sát ở góc chân nắp capo và mép rìa camera.
    """
    pts = cnt.reshape(-1, 2)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    # Ngưỡng 65% chiều cao vùng quan sát lòng đường (khoảng y > 590px)
    y_threshold = y_top + int((y_bottom - y_top) * 0.65)

    # 1. Góc dưới bên trái: áp sát mép trái (x <= 130) và chân nắp capo (y >= y_bottom - 25) mà không vươn lên lòng đường
    if x_min <= 130 and y_max >= (y_bottom - 25) and y_min >= y_threshold:
        return True

    # 2. Góc dưới bên phải: áp sát mép phải (x >= width - 130) và chân nắp capo (y >= y_bottom - 25) mà không vươn lên lòng đường
    if x_max >= (width - 130) and y_max >= (y_bottom - 25) and y_min >= y_threshold:
        return True

    return False


def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    iom = interArea / float(min(boxAArea, boxBArea) + 1e-6)
    return max(iou, iom * 0.5)


class EncroachTracker:
    """
    Theo dõi vết (Tracking) các vật thể lấn chiếm liên tục theo không gian (IoU) và thời gian (giây).
    Chỉ khi một vật thể được theo dõi và tồn tại ổn định ít nhất `min_duration_sec` (ví dụ 0.5 - 0.6 giây)
    mới chính thức xác nhận là lấn chiếm (tránh hoàn toàn hiện tượng nhấp nháy 0.2 - 0.4s do nhiễu/bóng xe,
    đồng thời phát hiện kịp thời từ xa khi xe đang di chuyển).
    """
    def __init__(self, fps=30.0, min_duration_sec=0.5, max_lost_sec=0.6):
        self.fps = max(1.0, float(fps))
        self.min_frames = max(3, int(min_duration_sec * self.fps))
        self.max_lost_frames = max(2, int(max_lost_sec * self.fps))
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        matched_tracks = set()
        matched_dets = set()

        for t_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = None
            for d_idx, det in enumerate(detections):
                if d_idx in matched_dets:
                    continue
                iou = bbox_iou(track["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = d_idx
            if best_iou > 0.15:
                matched_dets.add(best_det_idx)
                matched_tracks.add(t_id)
                det = detections[best_det_idx]
                track["bbox"] = det["bbox"]
                track["cnt"] = det["cnt"]
                track["area"] = det["area"]
                track["depth"] = det["depth"]
                track["yolo_pt"] = det.get("yolo_pt")
                track["seg_pt"] = det.get("seg_pt")
                track["side"] = det.get("side", "Trái")
                track["last_seen"] = frame_idx
                track["total_seen"] += 1
                if track["area"] > track["max_area"]:
                    track["max_area"] = det["area"]
                    track["max_depth"] = det["depth"]
                    track["peak_frame"] = frame_idx
                if track["total_seen"] >= self.min_frames:
                    track["confirmed"] = True

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "start_frame": frame_idx,
                    "start_time_sec": round(frame_idx / self.fps, 2),
                    "last_seen": frame_idx,
                    "total_seen": 1,
                    "confirmed": False,
                    "bbox": det["bbox"],
                    "cnt": det["cnt"],
                    "area": det["area"],
                    "depth": det["depth"],
                    "yolo_pt": det.get("yolo_pt"),
                    "seg_pt": det.get("seg_pt"),
                    "side": det.get("side", "Trái"),
                    "max_area": det["area"],
                    "max_depth": det["depth"],
                    "peak_frame": frame_idx,
                    "peak_frame_rgb": None,
                    "saved": False,
                }
                self.next_id += 1

        finished_tracks = []
        for t_id, track in list(self.tracks.items()):
            if frame_idx - track["last_seen"] > self.max_lost_frames:
                if track["confirmed"] and not track["saved"]:
                    finished_tracks.append(track)
                    track["saved"] = True
                del self.tracks[t_id]

        active_confirmed = [
            t for t in self.tracks.values()
            if t["confirmed"] and (frame_idx - t["last_seen"] <= 2)
        ]
        return active_confirmed, finished_tracks


class VendorTracker:
    """
    Theo dõi và phát hiện người bán hàng rong / buôn bán lấn chiếm lòng đường.
    Kết hợp phân đoạn ngữ nghĩa (person + sạp hàng/xe đạp/thùng hàng/biển báo) với tracking theo thời gian
    để xác nhận đối tượng buôn bán / hàng rong đứng tại chỗ gần mép đường.
    """

    def __init__(self, fps=30.0, min_stationary_sec=1.5, max_lost_sec=1.5):
        self.fps = max(1.0, float(fps))
        self.min_stationary_frames = max(3, int(min_stationary_sec * self.fps))
        self.max_lost_frames = max(2, int(max_lost_sec * self.fps))
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def find_candidates(
        pred, labels, corridor_mask, left_x_by_y, right_x_by_y, edge_thresh, h, w
    ):
        """
        Tìm ứng viên vendor: person đứng gần mép đường cong thực tế, có thể kèm xe đạp/xe đẩy/sạp hàng/biển báo.
        """
        if left_x_by_y is None or right_x_by_y is None:
            return []

        # Lấy người (person)
        person_ids = [k for k, v in labels.items() if v == "person"]
        person_mask_v = np.isin(pred, person_ids).astype(np.uint8) * 255
        person_mask_v = cv2.morphologyEx(
            person_mask_v, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )

        # Loại trừ người đang điều khiển phương tiện cơ giới (motorcycle, rider, car, bus, truck)
        motor_ids = [k for k, v in labels.items() if v in {"motorcycle", "rider", "car", "bus", "truck"}]
        motor_mask = np.isin(pred, motor_ids).astype(np.uint8) * 255
        motor_dil = cv2.dilate(motor_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

        # Dấu hiệu hàng hóa / phương tiện buôn bán: xe đạp (bicycle), biển hiệu/giá (traffic sign), thùng hàng/sọt (fence)
        goods_ids = [k for k, v in labels.items() if v in {"bicycle", "traffic sign", "fence"}]
        goods_mask = np.isin(pred, goods_ids).astype(np.uint8) * 255
        goods_cnts, _ = cv2.findContours(goods_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc bỏ các contour hạ tầng cố định quá lớn (hàng rào công viên, cổng sắt nhà dân)
        valid_goods_cnts = []
        for gc in goods_cnts:
            g_area = cv2.contourArea(gc)
            if g_area < 150:
                continue
            gx, gy, gw, gh = cv2.boundingRect(gc)
            if g_area > 35000 or gw > 350 or gh > 350:
                continue
            valid_goods_cnts.append(gc)

        person_cnts, _ = cv2.findContours(person_mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        y_bot_limit = int(h * 0.94)

        for p_cnt in person_cnts:
            p_area = cv2.contourArea(p_cnt)
            if p_area < 350:
                continue
            pm = cv2.moments(p_cnt)
            if pm["m00"] < 1:
                continue
            pcx = int(pm["m10"] / pm["m00"])
            pcy = int(pm["m01"] / pm["m00"])

            # Bỏ qua ngoài khung hình hoặc vùng nắp capo / gương xe máy bên dưới
            if pcy >= y_bot_limit or pcx >= w or pcy < int(h * 0.35):
                continue

            pbx, pby, pbw, pbh = cv2.boundingRect(p_cnt)

            # BỘ LỌC ĐÁM ĐÔNG / ÙN TẮC GIAO THÔNG (Traffic Jam / Dense Crowd Exclusion):
            # Người bán hàng rong đơn lẻ (kể cả kèm gánh hàng/sọt) có giới hạn kích thước vật lý:
            # Chiều rộng person đơn lẻ không quá 380px, chiều cao không quá 450px
            if pbw > 380 or pbh > 450:
                continue

            # Mật độ phương tiện cơ giới trong bounding box của person (ùn tắc nhiều xe máy/ô tô):
            # Nếu trong vùng bbox có quá nhiều pixel phương tiện cơ giới (motorcycle, rider, car, truck, bus)
            # thì đây là dòng xe cộ ùn tắc, không phải người bán hàng rong
            roi_pred = pred[pby:pby + pbh, pbx:pbx + pbw]
            motor_px = np.sum(np.isin(roi_pred, [12, 13, 14, 15, 17]))
            if motor_px > 2500 or (motor_px / max(1, pbw * pbh) > 0.05 and motor_px > 1200):
                continue

            # Loại trừ người đang lái xe máy / ngồi trong ô tô lưu thông
            p_roi = motor_dil[pby:pby + pbh, pbx:pbx + pbw]
            p_m_roi = person_mask_v[pby:pby + pbh, pbx:pbx + pbw]
            if np.sum((p_roi > 0) & (p_m_roi > 0)) / max(1, np.sum(p_m_roi > 0)) > 0.15:
                continue

            # Khoảng cách đến 2 đường biên cong thực tế
            left_x = float(left_x_by_y[pcy])
            right_x = float(right_x_by_y[pcy])

            # BỘ LỌC LÀN XE LƯU THÔNG (Traffic Lane Exclusion):
            # Tuyệt đối loại trừ người / phương tiện đang di chuyển ở giữa lòng đường xe chạy
            if pcx > (left_x + 60) and pcx < (right_x - 60):
                continue

            dl = abs(pcx - left_x)
            dr = abs(pcx - right_x)
            near_side = "Trái" if dl < dr else "Phải"
            edge_x = int(right_x if near_side == "Phải" else left_x)

            # Người bán hàng phải đứng ở mép đường/vỉa hè sát mép, không quá sâu vào sân/nhà dân
            if near_side == "Phải":
                if pcx < right_x - 50 or pcx > right_x + 300:
                    continue
            else:
                if pcx > left_x + 50 or pcx < left_x - 300:
                    continue

            # Gom cụm hàng hóa / sọt hoa quả / biển hiệu / xe đạp buôn bán kề sát về phía mép đường
            merged_cnt = p_cnt
            has_goods = False
            for gc in valid_goods_cnts:
                gx, gy, gw, gh = cv2.boundingRect(gc)
                if near_side == "Phải":
                    if gx < pbx + pbw * 0.8:
                        gap_x = max(0, pbx - (gx + gw))
                        gap_y = max(0, max(pby - (gy + gh), gy - (pby + pbh)))
                        if gap_x < 75 and gap_y < 75:
                            has_goods = True
                            merged_cnt = np.vstack([merged_cnt, gc])
                else:
                    if gx + gw > pbx + pbw * 0.2:
                        gap_x = max(0, gx - (pbx + pbw))
                        gap_y = max(0, max(pby - (gy + gh), gy - (pby + pbh)))
                        if gap_x < 75 and gap_y < 75:
                            has_goods = True
                            merged_cnt = np.vstack([merged_cnt, gc])

            bx, by, bw, bh = cv2.boundingRect(merged_cnt)

            # Kích thước tối thiểu của cụm người bán hàng rong (loại bỏ các đốm nhiễu li ti ở xa)
            if bw < 50 or bh < 60 or (bw * bh < 3500):
                continue

            # Chặn trên kích thước cụm hàng rong (loại bỏ dính chùm đám đông kẹt xe)
            if bw > 380 or bh > 450:
                continue

            # Kiểm tra vùng cụm merged_cnt: Tuyệt đối không được chứa ô tô, xe tải, xe buýt (car, truck, bus)
            # hoặc mật độ phương tiện cơ giới quá lớn (tránh nhận nhầm xe dừng đèn đỏ / ùn tắc giao thông)
            merged_roi = pred[by:by + bh, bx:bx + bw]
            car_truck_bus_px = np.sum(np.isin(merged_roi, [13, 14, 15]))
            if car_truck_bus_px > 1000:
                continue

            motor_merged_px = np.sum(np.isin(merged_roi, [12, 13, 14, 15, 17]))
            if motor_merged_px > 2500 or (motor_merged_px / max(1, bw * bh) > 0.25):
                continue

            # Tính độ sâu lấn đường của cụm hàng rong
            if near_side == "Phải":
                road_depth = max(0.0, float(edge_x - bx))
                dist_to_edge = max(0.0, float(bx - edge_x)) if bx > edge_x else 0.0
            else:
                road_depth = max(0.0, float((bx + bw) - edge_x))
                dist_to_edge = max(0.0, float(edge_x - (bx + bw))) if (bx + bw) < edge_x else 0.0

            # Ứng viên hàng rong phải có hàng hóa buôn bán (sạp, sọt, xe, biển) VÀ tiếp xúc/sát mép đường (<= 80px)
            if not has_goods or dist_to_edge > min(edge_thresh, 80.0):
                continue

            candidates.append({
                "centroid": (pcx, pcy),
                "bbox": (int(bx), int(by), int(bw), int(bh)),
                "area": float(p_area),
                "cnt": merged_cnt,
                "has_vehicle": has_goods,
                "has_goods": has_goods,
                "dist_to_edge": float(dist_to_edge),
                "road_depth": float(road_depth),
                "side": near_side,
                "edge_x": edge_x,
            })

        return candidates

    def update(self, candidates, frame_idx):
        """Cập nhật tracking vendor theo thời gian thực."""
        matched_tracks = set()
        matched_cands = set()

        for t_id, track in list(self.tracks.items()):
            best_score = 0.0
            best_idx = None
            for ci, c in enumerate(candidates):
                if ci in matched_cands:
                    continue
                iou = bbox_iou(track["bbox"], c["bbox"])
                d = np.sqrt(
                    (c["centroid"][0] - track["centroid"][0]) ** 2
                    + (c["centroid"][1] - track["centroid"][1]) ** 2
                )
                if iou > 0.05 or d < 140:
                    score = iou * 100 + (140 - min(140, d))
                    if score > best_score:
                        best_score = score
                        best_idx = ci

            if best_idx is not None:
                matched_cands.add(best_idx)
                matched_tracks.add(t_id)
                c = candidates[best_idx]
                track["centroid"] = c["centroid"]
                track["bbox"] = c["bbox"]
                track["area"] = c["area"]
                track["cnt"] = c["cnt"]
                track["has_vehicle"] = track.get("has_vehicle", False) or c.get("has_vehicle", False)
                track["side"] = c["side"]
                track["dist_to_edge"] = c["dist_to_edge"]
                track["road_depth"] = c["road_depth"]
                track["edge_x"] = c["edge_x"]
                track["last_seen"] = frame_idx
                track["total_seen"] += 1
                track.setdefault("positions", []).append(c["centroid"])

                # Xác nhận là hàng rong đứng bán khi quan sát ổn định tối thiểu 20-25 frames (>= 0.7s)
                if track["total_seen"] >= max(20, int(0.7 * self.fps)):
                    track["confirmed"] = True

        for ci, c in enumerate(candidates):
            if ci not in matched_cands:
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "type": "vendor",
                    "start_frame": frame_idx,
                    "start_time_sec": round(frame_idx / self.fps, 2),
                    "centroid": c["centroid"],
                    "bbox": c["bbox"],
                    "area": c["area"],
                    "cnt": c["cnt"],
                    "has_vehicle": c.get("has_vehicle", False),
                    "side": c["side"],
                    "dist_to_edge": c["dist_to_edge"],
                    "road_depth": c["road_depth"],
                    "edge_x": c["edge_x"],
                    "last_seen": frame_idx,
                    "total_seen": 1,
                    "confirmed": False,
                    "saved": False,
                    "peak_frame": frame_idx,
                    "peak_frame_rgb": None,
                    "positions": [c["centroid"]],
                    "drift": 0.0,
                    "is_moving": False,
                }
                self.next_id += 1

        finished = []
        for t_id, track in list(self.tracks.items()):
            if frame_idx - track["last_seen"] > self.max_lost_frames:
                if track["confirmed"] and not track["saved"]:
                    finished.append(track)
                    track["saved"] = True
                del self.tracks[t_id]

        active = [
            t for t in self.tracks.values()
            if t["confirmed"] and (frame_idx - t["last_seen"] <= 2)
        ]
        return active, finished


def fit_dynamic_road_lines(
    pred,
    h,
    w,
    road_id=0,
    sidewalk_id=1,
    prev_left=None,
    prev_right=None,
    alpha=0.15,
):
    """
    Trích xuất và khớp 2 đường biên mép đường real-time từ mặt nạ phân đoạn SegFormer.
    - Biên phải: Bám sát mép trong của vỉa hè phải (pred == sidewalk_id).
    - Biên trái: Bám sát mép ngoài của vỉa hè trái. Khi bị đống cát che khuất mép đường,
      tuyệt đối KHÔNG lấy theo biên cong đống cát mà tự động nội suy / giữ vững trajectory
      của mép đường thẳng từ các frame trước.
    - Làm mượt theo thời gian EMA để đường biên di chuyển liên tục, không bị rung giật.
    """
    sw = (pred == sidewalk_id).astype(np.uint8)

    y_top = int(h * 0.40)
    y_bot = int(h * 0.94)

    # 1. BIÊN PHẢI (RIGHT CURB): Mép trong của vỉa hè bên phải (giáp với mặt đường)
    right_pts = []
    for y in range(y_top, y_bot, 4):
        xs = np.where(sw[y, int(w * 0.45):] > 0)[0]
        if len(xs) > 6:
            curb_x = xs[0] + int(w * 0.45)
            right_pts.append((y, curb_x))

    cur_right = None
    if len(right_pts) >= 8:
        pts = np.array(right_pts, dtype=np.float32)
        best_r = None
        best_score = -1e9
        N = len(pts)
        for _ in range(60):
            idx = np.random.choice(N, 2, replace=False)
            y1, y2 = pts[idx, 0]
            x1, x2 = pts[idx, 1]
            if abs(y1 - y2) < 25:
                continue
            a = (x2 - x1) / (y2 - y1)
            if not (1.8 <= a <= 4.0):
                continue
            b = x1 - a * y1
            pred_x = a * pts[:, 0] + b
            diff = np.abs(pts[:, 1] - pred_x)
            inliers = diff <= 25
            num_in = np.sum(inliers)
            if num_in >= 6:
                score = num_in * 10 - np.std(pts[inliers, 1] - pred_x[inliers])
                if score > best_score:
                    best_score = score
                    poly = np.polyfit(pts[inliers, 0], pts[inliers, 1], 1)
                    best_r = (float(poly[0]), float(poly[1]))
        cur_right = best_r

    # 2. BIÊN TRÁI (LEFT CURB): Mép ngoài của vỉa hè bên trái (tiếp giáp mặt đường)
    left_pts = []
    for y in range(y_top, y_bot, 4):
        xs = np.where(sw[y, :int(w * 0.55)] > 0)[0]
        if len(xs) > 6:
            curb_x = xs[-1]
            # Loại bỏ các điểm nhảy sâu vào lòng đường (do đống cát) hoặc nhảy thụt sâu vào trong vỉa hè (do xe máy/người đi bộ che)
            if prev_left is not None:
                exp_x = prev_left[0] * y + prev_left[1]
                if abs(curb_x - exp_x) > 30:
                    continue
            left_pts.append((y, curb_x))

    cur_left = None
    # Chỉ khớp khi có đủ điểm vỉa hè không bị che khuất và trải dài theo chiều dọc
    if len(left_pts) >= 10:
        pts = np.array(left_pts, dtype=np.float32)
        y_span = pts[:, 0].max() - pts[:, 0].min()
        if y_span >= 60:
            best_l = None
            best_score = -1e9
            N = len(pts)
            for _ in range(60):
                idx = np.random.choice(N, 2, replace=False)
                y1, y2 = pts[idx, 0]
                x1, x2 = pts[idx, 1]
                if abs(y1 - y2) < 25:
                    continue
                a = (x2 - x1) / (y2 - y1)
                # Giới hạn góc nghiêng phối cảnh mép trái thực tế [-3.2, -1.75], tránh bị sụp góc thẳng đứng vào vỉa hè
                if not (-3.2 <= a <= -1.75):
                    continue
                b = x1 - a * y1
                pred_x = a * pts[:, 0] + b
                diff = np.abs(pts[:, 1] - pred_x)
                inliers = diff <= 20
                num_in = np.sum(inliers)
                if num_in >= 6:
                    score = num_in * 10 - np.std(pts[inliers, 1] - pred_x[inliers])
                    if score > best_score:
                        best_score = score
                        poly = np.polyfit(pts[inliers, 0], pts[inliers, 1], 1)
                        best_l = (float(poly[0]), float(poly[1]))
            cur_left = best_l

    # Cập nhật EMA hoặc NỘI SUY từ các frame trước khi mép bị đống cát che
    if cur_left is not None:
        if prev_left is not None:
            la = alpha * cur_left[0] + (1 - alpha) * prev_left[0]
            lb = alpha * cur_left[1] + (1 - alpha) * prev_left[1]
        else:
            la, lb = cur_left
    elif prev_left is not None:
        la, lb = prev_left
    else:
        # Fallback an toàn trên mép vỉa hè trái theo tỷ lệ màn hình
        la = (0.25 * w - 0.08 * w) / (y_top - y_bot)
        lb = 0.08 * w - la * y_bot

    if cur_right is not None:
        if prev_right is not None:
            ra = alpha * cur_right[0] + (1 - alpha) * prev_right[0]
            rb = alpha * cur_right[1] + (1 - alpha) * prev_right[1]
        else:
            ra, rb = cur_right
    elif prev_right is not None:
        ra, rb = prev_right
    else:
        ra = (0.75 * w - 0.92 * w) / (y_top - y_bot)
        rb = 0.92 * w - ra * y_bot

    return (la, lb), (ra, rb)


def fit_dynamic_road_lines_yolo(
    yolo_pred,
    h,
    w,
    road_id=0,
    sidewalk_id=1,
    prev_left=None,
    prev_right=None,
    alpha=0.15,
):
    """
    Trích xuất và khớp 2 đường biên mép đường siêu ổn định từ mặt nạ YOLO Semantic Segmentation.
    YOLO có inductive bias trơn láng của CNN nên đường biên cực kỳ phẳng và thẳng,
    không bị lừa bởi đống cát (coi cát là mặt đường).
    - Biên trái: Mép ngoài cùng bên trái của mặt đường YOLO (road_id == 0).
    - Biên phải: Mép trong của vỉa hè phải hoặc mép ngoài cùng bên phải của mặt đường.
    - Áp dụng RANSAC & EMA smoothing để đường biên di chuyển mượt mà tuyệt đối.
    """
    road_mask = (yolo_pred == road_id).astype(np.uint8)
    sw_mask = (yolo_pred == sidewalk_id).astype(np.uint8)
    # Loại trừ các pixel thuộc phương tiện giao thông để mép đường không bị bắt nhầm vào sườn xe
    veh_mask = np.isin(yolo_pred, [11, 12, 13, 14, 15, 17, 18]).astype(np.uint8)
    veh_dilated = cv2.dilate(veh_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

    y_top = int(h * 0.40)
    y_bot = int(h * 0.94)

    # 1. BIÊN PHẢI (RIGHT CURB)
    right_pts = []
    for y in range(y_top, y_bot, 4):
        v_xs = np.where(veh_mask[y] > 0)[0]
        right_min_x = max(int(w * 0.45), v_xs[-1] if len(v_xs) > 0 else int(w * 0.45))

        sw_xs = np.where(sw_mask[y, int(w * 0.45):] > 0)[0]
        if len(sw_xs) > 6:
            curb_x = sw_xs[0] + int(w * 0.45)
            if prev_right is not None:
                exp_x = prev_right[0] * y + prev_right[1]
                if abs(curb_x - exp_x) > (w * 0.12):
                    continue
            right_pts.append((y, curb_x))
        else:
            road_xs = np.where((road_mask[y, right_min_x:] > 0) & (veh_dilated[y, right_min_x:] == 0))[0]
            if len(road_xs) > 6:
                curb_x = road_xs[-1] + right_min_x
                if prev_right is not None:
                    exp_x = prev_right[0] * y + prev_right[1]
                    if abs(curb_x - exp_x) > (w * 0.12):
                        continue
                right_pts.append((y, curb_x))

    cur_right = None
    if len(right_pts) >= 8:
        pts = np.array(right_pts, dtype=np.float32)
        best_r = None
        best_score = -1e9
        N = len(pts)
        for _ in range(120):
            idx = np.random.choice(N, 2, replace=False)
            y1, y2 = pts[idx, 0]
            x1, x2 = pts[idx, 1]
            if abs(y1 - y2) < 20:
                continue
            a = (x2 - x1) / (y2 - y1)
            if not (0.4 <= a <= 5.0):
                continue
            b = x1 - a * y1
            # Ràng buộc đường phải: không cắt vào trung tâm lòng đường
            if (a * y_bot + b) < (w * 0.55):
                continue
            if (a * y_top + b) < (w * 0.45):
                continue
            pred_x = a * pts[:, 0] + b
            diff = np.abs(pts[:, 1] - pred_x)
            inliers = diff <= (w * 0.025)
            num_in = np.sum(inliers)
            if num_in >= 6:
                score = num_in * 10 - np.std(pts[inliers, 1] - pred_x[inliers])
                if score > best_score:
                    poly = np.polyfit(pts[inliers, 0], pts[inliers, 1], 1)
                    pa, pb = float(poly[0]), float(poly[1])
                    if not (0.4 <= pa <= 5.0):
                        continue
                    if (pa * y_bot + pb) < (w * 0.55):
                        continue
                    if (pa * y_top + pb) < (w * 0.45):
                        continue
                    best_score = score
                    best_r = (pa, pb)
        cur_right = best_r

    # 2. BIÊN TRÁI (LEFT CURB)
    left_pts = []
    for y in range(y_top, y_bot, 4):
        # Mép trái tuyệt đối không được vượt sang phía bên phải của bất kỳ phương tiện giao thông nào trên scanline
        v_xs = np.where(veh_mask[y] > 0)[0]
        left_max_x = min(int(w * 0.55), v_xs[0]) if (len(v_xs) > 0 and v_xs[0] > 80) else int(w * 0.55)

        # Ưu tiên mép trong cùng của dải phân cách/vỉa hè trái (sw_mask) giáp với lòng đường:
        sw_xs = np.where(sw_mask[y, :left_max_x] > 0)[0]
        if len(sw_xs) > 6:
            curb_x = sw_xs[-1]
            if prev_left is not None:
                exp_x = prev_left[0] * y + prev_left[1]
                if 0 <= exp_x <= (w * 0.50) and abs(curb_x - exp_x) > (w * 0.15):
                    continue
            left_pts.append((y, curb_x))
        else:
            road_xs = np.where((road_mask[y, :left_max_x] > 0) & (veh_dilated[y, :left_max_x] == 0))[0]
            if len(road_xs) > 6:
                curb_x = road_xs[0]
                if curb_x < int(w * 0.04):
                    continue
                if prev_left is not None:
                    exp_x = prev_left[0] * y + prev_left[1]
                    if 0 <= exp_x <= (w * 0.50) and abs(curb_x - exp_x) > (w * 0.15):
                        continue
                left_pts.append((y, curb_x))

    cur_left = None
    if len(left_pts) >= 8:
        pts = np.array(left_pts, dtype=np.float32)
        best_l = None
        best_score = -1e9
        N = len(pts)
        for _ in range(120):
            idx = np.random.choice(N, 2, replace=False)
            y1, y2 = pts[idx, 0]
            x1, x2 = pts[idx, 1]
            if abs(y1 - y2) < 20:
                continue
            a = (x2 - x1) / (y2 - y1)
            if not (-5.0 <= a <= -0.4):
                continue
            b = x1 - a * y1
            # Ràng buộc đường trái: không cắt sâu vào lòng đường và không sụp ra quá xa
            if (a * y_bot + b) > (w * 0.35) or (a * y_bot + b) < (-w * 0.20):
                continue
            if (a * y_top + b) > (w * 0.45) or (a * y_top + b) < (w * 0.05):
                continue
            pred_x = a * pts[:, 0] + b
            diff = np.abs(pts[:, 1] - pred_x)
            inliers = diff <= (w * 0.03)
            num_in = np.sum(inliers)
            if num_in >= 6:
                score = num_in * 10 - np.std(pts[inliers, 1] - pred_x[inliers])
                if score > best_score:
                    poly = np.polyfit(pts[inliers, 0], pts[inliers, 1], 1)
                    pa, pb = float(poly[0]), float(poly[1])
                    if not (-5.0 <= pa <= -0.4):
                        continue
                    if (pa * y_bot + pb) > (w * 0.35) or (pa * y_bot + pb) < (-w * 0.20):
                        continue
                    if (pa * y_top + pb) > (w * 0.45) or (pa * y_top + pb) < (w * 0.05):
                        continue
                    best_score = score
                    best_l = (pa, pb)
        cur_left = best_l

    # Cập nhật EMA (hoặc gán trực tiếp nếu chưa có prev)
    if cur_left is not None:
        if prev_left is not None:
            la = alpha * cur_left[0] + (1 - alpha) * prev_left[0]
            lb = alpha * cur_left[1] + (1 - alpha) * prev_left[1]
        else:
            la, lb = cur_left
    elif prev_left is not None:
        la, lb = prev_left
    else:
        # Fallback an toàn trên mép vỉa hè trái thay vì lệch âm ra ngoài màn hình
        la = (0.25 * w - 0.08 * w) / (y_top - y_bot)
        lb = 0.08 * w - la * y_bot

    if cur_right is not None:
        if prev_right is not None:
            ra = alpha * cur_right[0] + (1 - alpha) * prev_right[0]
            rb = alpha * cur_right[1] + (1 - alpha) * prev_right[1]
        else:
            ra, rb = cur_right
    elif prev_right is not None:
        ra, rb = prev_right
    else:
        ra = (0.75 * w - 0.92 * w) / (y_top - y_bot)
        rb = 0.92 * w - ra * y_bot

    return (la, lb), (ra, rb)


def extract_curved_road_corridor(
    mask_road,
    mask_sw,
    veh_mask,
    h,
    w,
    y_top=None,
    y_bot=None,
    prev_left_curve=None,
    prev_right_curve=None,
    alpha=0.30,
    yolo_lines=None,
):
    """
    Trích xuất hành lang lòng đường chuẩn bám khớp 100% theo 2 đường biên tự nhiên của road_mask.mp4.
    - Lấy trực tiếp biên mép trái (xs[0]) và mép phải (xs[-1]) từ mặt nạ road_mask của SegFormer.
    - Tự động khắc phục hiện tượng thắt mép do đống cát / vật thể lạ bằng đường biên chuẩn YOLO RANSAC:
      Khi mép đường bị vật thể lạ chiếm đóng (không phải vỉa hè), mép hành lang được mở rộng theo mép vỉa hè thật.
    - Gộp xe cộ trên đường bằng cách tô kín toàn bộ khoảng trống ngang giữa mép trái và mép phải trên từng scanline.
    - Đảm bảo 100% không còn lỗ thủng xe màu đen và 2 đường biên 2 bên uốn lượn mượt mà tuyệt đối.
    """
    if y_top is None:
        y_top = int(h * 0.35)
    if y_bot is None:
        y_bot = int(h * 0.98)

    # 1. Lọc nhiễu li ti ngoài lòng đường của mask_road
    cnts, _ = cv2.findContours(mask_road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_road = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        max_area = max(cv2.contourArea(c) for c in cnts)
        for c in cnts:
            if cv2.contourArea(c) > max(1000.0, max_area * 0.05):
                cv2.drawContours(clean_road, [c], -1, 255, -1)

    all_ys = np.arange(y_top, y_bot + 1)
    left_x = np.full(h, -1.0, dtype=np.float32)
    right_x = np.full(h, -1.0, dtype=np.float32)

    for y in all_ys:
        xs = np.where(clean_road[y] > 0)[0]
        if len(xs) > 0:
            left_x[y] = float(xs[0])
            right_x[y] = float(xs[-1])
        else:
            xs_v = np.where((veh_mask[y] > 0) & (~mask_sw[y]))[0]
            if len(xs_v) > 0:
                left_x[y] = float(xs_v[0])
                right_x[y] = float(xs_v[-1])

    valid_ys = np.where(left_x >= 0)[0]
    valid_ys = valid_ys[(valid_ys >= y_top) & (valid_ys <= y_bot)]

    if len(valid_ys) >= 5:
        interp_left = np.interp(all_ys, valid_ys, left_x[valid_ys])
        interp_right = np.interp(all_ys, valid_ys, right_x[valid_ys])
    elif yolo_lines is not None:
        (la, lb), (ra, rb) = yolo_lines
        interp_left = np.clip(la * all_ys + lb, 0, w - 1)
        interp_right = np.clip(ra * all_ys + rb, 0, w - 1)
    else:
        interp_left = np.zeros(len(all_ys), dtype=np.float32)
        interp_right = np.full(len(all_ys), w - 1, dtype=np.float32)

    # 4. Cầu nối vượt qua dị vật (đống cát, chướng ngại vật chiếm lòng đường) bằng mép vỉa hè chuẩn YOLO
    if yolo_lines is not None:
        (la, lb), (ra, rb) = yolo_lines
        for idx, y in enumerate(all_ys):
            yl = la * y + lb
            cl = interp_left[idx]
            if yl < cl - 15:
                x1 = max(0, int(yl))
                x2 = min(w, int(cl))
                if x2 > x1:
                    # Nếu khoảng trống giữa mép YOLO và mép nhựa đường không phải vỉa hè -> Dị vật chiếm lòng đường
                    if np.mean(mask_sw[y, x1:x2]) < 0.30:
                        interp_left[idx] = yl

            yr = ra * y + rb
            cr = interp_right[idx]
            if yr > cr + 15:
                x1 = max(0, int(cr))
                x2 = min(w, int(yr))
                if x2 > x1:
                    if np.mean(mask_sw[y, x1:x2]) < 0.30:
                        interp_right[idx] = yr

    # 5. Làm mượt không gian bằng bộ lọc Gaussian biên (ksize=61, sigma=18)
    # Loại bỏ triệt để hiện tượng nhấp nhô răng cưa pixel của SegFormer, tạo đường biên uốn lượn mượt mà như Ảnh 2
    ksize = 61
    sigma = 18.0
    kg = cv2.getGaussianKernel(ksize, sigma).flatten()
    half = ksize // 2
    pad_l = np.pad(interp_left, half, mode="edge")
    pad_r = np.pad(interp_right, half, mode="edge")
    smooth_l = np.convolve(pad_l, kg, mode="valid")
    smooth_r = np.convolve(pad_r, kg, mode="valid")

    # 6. Làm mượt thời gian EMA để đường biên ổn định không bị rung giật giữa các frame
    if prev_left_curve is not None and len(prev_left_curve) == len(smooth_l):
        smooth_l = alpha * smooth_l + (1.0 - alpha) * prev_left_curve
    if prev_right_curve is not None and len(prev_right_curve) == len(smooth_r):
        smooth_r = alpha * smooth_r + (1.0 - alpha) * prev_right_curve

    # Đảm bảo khoảng cách lòng đường thực tế trên all_ys tối thiểu 20px
    smooth_l = np.minimum(smooth_l, smooth_r - 20.0)

    # 7. Đỉnh đường chân trời kết thúc phẳng ngang tự nhiên theo thực tế (không vuốt nhọn thành hình chóp)
    full_ys = all_ys
    full_l = smooth_l
    full_r = smooth_r

    # 8. Tạo đa giác hành lang lòng đường mượt mà, đặc kín
    pts_l = np.column_stack([np.clip(full_l, 0, w - 1).astype(np.int32), full_ys])
    pts_r = np.column_stack([np.clip(full_r, 0, w - 1).astype(np.int32), full_ys])
    poly = np.vstack([pts_l, pts_r[::-1]])

    legal_corridor_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(legal_corridor_poly, [poly], 255)

    # 9. Bảng tra tọa độ x cho bất kỳ tung độ y nào (0 đến h-1)
    left_x_by_y = np.zeros(h, dtype=np.float32)
    right_x_by_y = np.full(h, w - 1, dtype=np.float32)
    left_x_by_y[:y_top] = full_l[0]
    left_x_by_y[y_top:y_bot + 1] = full_l
    left_x_by_y[y_bot + 1:] = full_l[-1]

    right_x_by_y[:y_top] = full_r[0]
    right_x_by_y[y_top:y_bot + 1] = full_r
    right_x_by_y[y_bot + 1:] = full_r[-1]

    return legal_corridor_poly, pts_l, pts_r, left_x_by_y, right_x_by_y, smooth_l, smooth_r


def get_solid_obstacle_mask(
    track,
    out_height,
    out_width,
    left_x_by_y=None,
    right_x_by_y=None,
):
    """
    Tạo mặt nạ nhị phân đặc kín (solid mask) cho vật thể lấn chiếm:
    - Bao phủ toàn bộ diện tích vật thể từ đường biên chuẩn (mép vỉa hè) đến đỉnh lấn sâu nhất vào lòng đường.
    - Đảm bảo 100% không bị rỗng ở giữa, không chỉ vẽ một đường viền mép mà tô đặc hoàn toàn diện tích vật thể.
    """
    cnt = track.get("cnt")
    if cnt is None:
        return np.zeros((out_height, out_width), dtype=np.uint8)

    obs_mask = np.zeros((out_height, out_width), dtype=np.uint8)
    cv2.drawContours(obs_mask, [cnt], -1, 255, -1)

    pts = cnt.reshape(-1, 2)
    if len(pts) == 0:
        return obs_mask

    y_min = int(pts[:, 1].min())
    y_max = int(pts[:, 1].max())

    side = track.get("side", "Trái")
    if track.get("yolo_pt") is not None and left_x_by_y is not None:
        edge_x = track["yolo_pt"][0]
        side = "Trái" if edge_x < (out_width / 2) else "Phải"

    unique_ys = np.unique(pts[:, 1])
    if len(unique_ys) > 1:
        interp_ys = np.arange(y_min, y_max + 1)
        if side in ("Trái", "left", "trai", "Trai"):
            max_xs_at_y = np.array([pts[pts[:, 1] == uy, 0].max() for uy in unique_ys])
            interp_max_xs = np.interp(interp_ys, unique_ys, max_xs_at_y).astype(int)
            for y, rx in zip(interp_ys, interp_max_xs):
                lx = int(left_x_by_y[y]) if left_x_by_y is not None else 0
                if rx >= lx:
                    obs_mask[y, max(0, lx):min(out_width, rx + 1)] = 255
        else:
            min_xs_at_y = np.array([pts[pts[:, 1] == uy, 0].min() for uy in unique_ys])
            interp_min_xs = np.interp(interp_ys, unique_ys, min_xs_at_y).astype(int)
            for y, lx in zip(interp_ys, interp_min_xs):
                rx = int(right_x_by_y[y]) if right_x_by_y is not None else (out_width - 1)
                if rx >= lx:
                    obs_mask[y, max(0, lx):min(out_width, rx + 1)] = 255

    return obs_mask


def main():
    args = parse_args()

    video_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    if not (0 < args.scale <= 1.0):
        raise ValueError("--scale phải nằm trong khoảng (0, 1.0]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(args.fp16 and device.type == "cuda")

    print(f"[INFO] Thiết bị: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Chế độ FP16: {'BẬT (Autocast)' if use_fp16 else 'TẮT (FP32)'}")

    mode = args.mode.lower().strip()
    yolo_model = None
    seg_model = None
    processor = None

    if mode == "hybrid":
        if not HAS_ULTRALYTICS:
            raise ImportError("Vui lòng cài đặt ultralytics để dùng chế độ Hybrid: pip install ultralytics")
        print(f"[INFO] Chế độ hoạt động: HYBRID (YOLO kẻ biên chuẩn + SegFormer tách dị vật)")
        print(f"[INFO] 1. Tải YOLO Semantic Model ({args.yolo_model}) để kẻ đường biên chuẩn...")
        yolo_model = YOLO(args.yolo_model)
        
        seg_model_key = args.model.lower().strip()
        seg_model_name = MODEL_CHOICES.get(seg_model_key, args.model)
        if seg_model_name.endswith(".pt") or "yolo" in seg_model_name.lower():
            seg_model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        print(f"[INFO] 2. Tải SegFormer Model ({seg_model_name}) để phân đoạn mặt đường & phát hiện dị vật...")
        processor = AutoImageProcessor.from_pretrained(seg_model_name)
        seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name).to(device)
        seg_model.eval()
        labels = {int(k): str(v).lower() for k, v in seg_model.config.id2label.items()}
    elif mode == "yolo":
        if not HAS_ULTRALYTICS:
            raise ImportError("Vui lòng cài đặt ultralytics: pip install ultralytics")
        model_key = args.model.lower().strip()
        model_name = MODEL_CHOICES.get(model_key, args.model)
        if not model_name.endswith(".pt") and "yolo" not in model_name:
            model_name = "yolo26n-sem.pt"
        print(f"[INFO] Chế độ hoạt động: STANDALONE YOLO ({model_name})")
        yolo_model = YOLO(model_name)
        labels = {int(k): str(v).lower() for k, v in yolo_model.names.items()}
    else:  # segformer
        seg_model_key = args.model.lower().strip()
        seg_model_name = MODEL_CHOICES.get(seg_model_key, args.model)
        if seg_model_name.endswith(".pt") or "yolo" in seg_model_name.lower():
            seg_model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        print(f"[INFO] Chế độ hoạt động: STANDALONE SEGFORMER ({seg_model_name})")
        processor = AutoImageProcessor.from_pretrained(seg_model_name)
        seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name).to(device)
        seg_model.eval()
        labels = {int(k): str(v).lower() for k, v in seg_model.config.id2label.items()}

    road_class_ids = [k for k, v in labels.items() if v == "road"]
    sidewalk_class_ids = [k for k, v in labels.items() if v == "sidewalk"]
    corridor_class_ids = [k for k, v in labels.items() if v in CORRIDOR_CLASSES]

    road_id = road_class_ids[0] if road_class_ids else 0
    sidewalk_id = sidewalk_class_ids[0] if sidewalk_class_ids else 1

    # Hệ số đường thẳng mép đường chuẩn
    la, lb = parse_line_params(args.left_line, None, None)
    ra, rb = parse_line_params(args.right_line, None, None)

    prev_line_left = (la, lb) if la is not None else None
    prev_line_right = (ra, rb) if ra is not None else None

    if args.dynamic_lines:
        print(f"[INFO] Chế độ đường biên: REAL-TIME DYNAMIC (EMA smooth={args.line_smooth})")
        if prev_line_left and prev_line_right:
            print(f"[INFO] Hệ số khởi tạo ban đầu: Trái (x={la:.2f}*y+{lb:.0f}), Phải (x={ra:.2f}*y+{rb:.0f})")
        else:
            print(f"[INFO] Hệ số đường biên: TỰ ĐỘNG NHẬN DIỆN & KHỚP THEO VIDEO TRÊN FRAME 1")
    else:
        print(f"[INFO] Chế độ đường biên: CỐ ĐỊNH (Static)")
        if prev_line_left and prev_line_right:
            print(f"[INFO] Đường thẳng mép TRÁI chuẩn: x = {la:.3f} * y + {lb:.1f}")
            print(f"[INFO] Đường thẳng mép PHẢI chuẩn: x = {ra:.3f} * y + {rb:.1f}")
        else:
            print(f"[INFO] Đường thẳng mép chuẩn: TỰ ĐỘNG KHỚP TRÊN FRAME ĐẦU TIÊN")

    # Mở video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = max(1, int(width * args.scale))
    out_height = max(1, int(height * args.scale))
    out_fps = src_fps / max(1, args.skip)

    print(f"[INFO] Video input: {width}x{height} @ {src_fps:.2f} FPS | Tổng: {total_frames} frames")
    print(f"[INFO] Kích thước xuất: {out_width}x{out_height} @ {out_fps:.2f} FPS (skip={args.skip})")

    overlay_path = output_dir / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, out_fps, (out_width, out_height))

    road_writer = None
    encroach_writer = None
    if args.save_masks:
        road_path = output_dir / "road_mask.mp4"
        encroach_path = output_dir / "encroach_mask.mp4"
        road_writer = cv2.VideoWriter(str(road_path), fourcc, out_fps, (out_width, out_height))
        encroach_writer = cv2.VideoWriter(str(encroach_path), fourcc, out_fps, (out_width, out_height))

    if not overlay_writer.isOpened():
        cap.release()
        raise RuntimeError("Không tạo được VideoWriter đầu ra. Vui lòng kiểm tra codec OpenCV.")

    line_color_rgb = PALETTE.get(args.boundary_color.lower(), (255, 255, 0))

    # Khoảng chiều dọc xác định lòng đường (từ chân trời đến sát đáy màn hình)
    y_top = int(out_height * 0.35)
    y_bottom = int(out_height * 0.98)

    # Đa giác hành lang lòng đường chuẩn (Legal Corridor Polygon)
    ys_all = np.arange(out_height)
    if la is not None and ra is not None:
        legal_corridor_poly, (pt_left_top, pt_left_bot), (pt_right_top, pt_right_bot) = (
            compute_corridor_polygon(
                la, lb, ra, rb, y_top, y_bottom, out_width, out_height
            )
        )
        left_x_by_y = np.clip(la * ys_all + lb, 0, out_width - 1).astype(np.float32)
        right_x_by_y = np.clip(ra * ys_all + rb, 0, out_width - 1).astype(np.float32)
        pts_left_curve = np.column_stack([left_x_by_y[y_top:y_bottom].astype(np.int32), ys_all[y_top:y_bottom]])
        pts_right_curve = np.column_stack([right_x_by_y[y_top:y_bottom].astype(np.int32), ys_all[y_top:y_bottom]])
    else:
        legal_corridor_poly = np.zeros((out_height, out_width), dtype=np.uint8)
        pt_left_top, pt_left_bot = (0, y_top), (0, y_bottom)
        pt_right_top, pt_right_bot = (out_width - 1, y_top), (out_width - 1, y_bottom)
        left_x_by_y = np.zeros(out_height, dtype=np.float32)
        right_x_by_y = np.full(out_height, out_width - 1, dtype=np.float32)
        pts_left_curve = None
        pts_right_curve = None

    prev_curve_left = None
    prev_curve_right = None
    prev_yolo_l = None
    prev_yolo_r = None

    prev_probs = None
    frame_idx = 0
    processed_count = 0
    start_time = time.perf_counter()

    # Khởi tạo bộ ghi nhận sự việc (Incident Evidence Logger)
    incidents_dir = output_dir / "incidents"
    if args.save_incidents:
        incidents_dir.mkdir(parents=True, exist_ok=True)
    incidents_log = []
    active_incident = None

    # Khởi tạo bộ theo dõi vật thể lấn chiếm (Encroachment Tracker theo thời gian thực)
    tracker = EncroachTracker(fps=out_fps, min_duration_sec=args.min_duration)
    is_alarm_active = False

    # Khởi tạo bộ theo dõi hàng rong / buôn bán lấn chiếm mép đường
    vendor_tracker = (
        VendorTracker(fps=out_fps, min_stationary_sec=args.vendor_min_sec)
        if args.vendor_detect
        else None
    )

    try:
        while True:
            if args.max_frames > 0 and processed_count >= args.max_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.skip > 1 and (frame_idx % args.skip != 0):
                continue

            if args.scale != 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr, (out_width, out_height), interpolation=cv2.INTER_LINEAR
                )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            if mode == "hybrid":
                # 1. Inference với Ultralytics YOLO Semantic Segmentation để trích xuất đường biên chuẩn
                yolo_h = max(32, (out_height + 31) // 32 * 32)
                yolo_w = max(32, (out_width + 31) // 32 * 32)
                yolo_res = yolo_model(
                    frame_bgr,
                    imgsz=(yolo_h, yolo_w),
                    device=0 if device.type == "cuda" else "cpu",
                    verbose=False,
                )[0]
                if yolo_res.semantic_mask is not None:
                    yolo_pred_raw = yolo_res.semantic_mask.data.cpu().numpy()
                    if yolo_pred_raw.shape != (out_height, out_width):
                        yolo_pred = cv2.resize(yolo_pred_raw, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
                    else:
                        yolo_pred = yolo_pred_raw
                else:
                    yolo_pred = np.zeros((out_height, out_width), dtype=np.int32)

                # 2. Inference với SegFormer (Transformers) để phân đoạn mặt đường & tách dị vật
                image = Image.fromarray(frame_rgb)
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    if use_fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = seg_model(**inputs)
                    else:
                        outputs = seg_model(**inputs)

                    logits = F.interpolate(
                        outputs.logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )

                    probs = torch.softmax(logits[0], dim=0)

                    # Temporal smoothing khử nhấp nháy
                    if args.smooth > 0.0 and prev_probs is not None:
                        probs = args.smooth * probs + (1.0 - args.smooth) * prev_probs
                    prev_probs = probs.detach()

                    # Spatial smoothing trên bản đồ xác suất mặt đường
                    if args.spatial_smooth:
                        road_prob = probs[road_id].float().cpu().numpy()
                        road_prob_smooth = cv2.GaussianBlur(road_prob, (7, 7), 1.5)
                        probs[road_id] = torch.from_numpy(road_prob_smooth).to(probs.device, dtype=probs.dtype)

                    if args.conf_thresh > 0.0:
                        max_prob, pred_tensor = torch.max(probs, dim=0)
                        pred_tensor[max_prob < args.conf_thresh] = -1
                        pred = pred_tensor.cpu().numpy()
                    else:
                        pred = probs.argmax(dim=0).cpu().numpy()

            elif mode == "yolo":
                yolo_h = max(32, (out_height + 31) // 32 * 32)
                yolo_w = max(32, (out_width + 31) // 32 * 32)
                yolo_res = yolo_model(
                    frame_bgr,
                    imgsz=(yolo_h, yolo_w),
                    device=0 if device.type == "cuda" else "cpu",
                    verbose=False,
                )[0]
                if yolo_res.semantic_mask is not None:
                    pred_raw = yolo_res.semantic_mask.data.cpu().numpy()
                    if pred_raw.shape != (out_height, out_width):
                        pred = cv2.resize(pred_raw, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
                    else:
                        pred = pred_raw
                else:
                    pred = np.zeros((out_height, out_width), dtype=np.int32)
                yolo_pred = None
            else:  # Standalone SegFormer
                image = Image.fromarray(frame_rgb)
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    if use_fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = seg_model(**inputs)
                    else:
                        outputs = seg_model(**inputs)

                    logits = F.interpolate(
                        outputs.logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probs = torch.softmax(logits[0], dim=0)
                    if args.smooth > 0.0 and prev_probs is not None:
                        probs = args.smooth * probs + (1.0 - args.smooth) * prev_probs
                    prev_probs = probs.detach()

                    if args.spatial_smooth:
                        road_prob = probs[road_id].float().cpu().numpy()
                        road_prob_smooth = cv2.GaussianBlur(road_prob, (7, 7), 1.5)
                        probs[road_id] = torch.from_numpy(road_prob_smooth).to(probs.device, dtype=probs.dtype)

                    if args.conf_thresh > 0.0:
                        max_prob, pred_tensor = torch.max(probs, dim=0)
                        pred_tensor[max_prob < args.conf_thresh] = -1
                        pred = pred_tensor.cpu().numpy()
                    else:
                        pred = probs.argmax(dim=0).cpu().numpy()
                yolo_pred = None

            # 3. Mặt nạ vỉa hè (dual-model), mặt đường & phương tiện cơ giới trên đường
            if mode == "hybrid" and yolo_pred is not None:
                mask_sw = (pred == sidewalk_id) | (yolo_pred == sidewalk_id)
            else:
                mask_sw = (pred == sidewalk_id)
            mask_road = (pred == road_id) & (~mask_sw)
            # Gộp toàn bộ phương tiện giao thông và người điều khiển (person, rider, car, truck, bus, motorcycle, bicycle)
            # để đảm bảo toàn bộ xe cộ trên đường đều thuộc lòng đường, không bị lỗ thủng xe màu đen.
            # Việc loại trừ nhô lên trên bầu trời/cây cối được xử lý triệt để bằng cắt đường chân trời y_top_road
            # và giới hạn trong dải đa giác mép đường smooth_l, smooth_r.
            veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)

            # 4. Trích xuất hành lang lòng đường cong thích ứng & 2 đường biên uốn lượn
            if args.dynamic_lines:
                yolo_lines = None
                if mode == "hybrid" and yolo_pred is not None:
                    (la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(
                        yolo_pred,
                        out_height,
                        out_width,
                        road_id=road_id,
                        sidewalk_id=sidewalk_id,
                        prev_left=prev_yolo_l,
                        prev_right=prev_yolo_r,
                        alpha=args.line_smooth,
                    )
                    prev_yolo_l, prev_yolo_r = (la, lb), (ra, rb)
                    yolo_lines = ((la, lb), (ra, rb))

                (
                    legal_corridor_poly,
                    pts_left_curve,
                    pts_right_curve,
                    left_x_by_y,
                    right_x_by_y,
                    prev_curve_left,
                    prev_curve_right,
                ) = extract_curved_road_corridor(
                    mask_road,
                    mask_sw,
                    veh_mask,
                    out_height,
                    out_width,
                    y_top=y_top,
                    y_bot=y_bottom,
                    prev_left_curve=prev_curve_left,
                    prev_right_curve=prev_curve_right,
                    alpha=args.line_smooth,
                    yolo_lines=yolo_lines,
                )

            # 4. Mặt nạ hành lang giao thông & mặt đường chuẩn (road + sidewalk + traffic)
            normal_mask = np.isin(
                pred,
                [road_id, sidewalk_id]
                + [k for k, v in labels.items() if v in CORRIDOR_CLASSES]
            ).astype(np.uint8) * 255
            normal_mask = cv2.morphologyEx(
                normal_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            )

            # Mặt nạ lòng đường (road) mở rộng để kiểm tra tiếp giáp (Road Adjacency)
            road_mask_pure = (pred == road_id).astype(np.uint8)
            road_dilated = cv2.dilate(
                road_mask_pure,
                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
            )

            # 5. Xác định Vùng Lấn Chiếm (Encroachment Region)
            # Vùng nằm BÊN TRONG hành lang lòng đường chuẩn NHƯNG không thuộc normal_mask (dị vật: đống cát...)
            # VÀ PHẢI TIẾP XÚC / NẰM TRÊN MẶT LÒNG ĐƯỜNG (loại trừ hoàn toàn dị vật trên cây/bầu trời/vỉa hè)
            encroach_raw = (legal_corridor_poly == 255) & (normal_mask == 0) & (road_dilated > 0)
            encroach_mask = encroach_raw.astype(np.uint8) * 255
            encroach_mask = cv2.morphologyEx(
                encroach_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            )

            # Tìm các contour của vật thể lấn chiếm
            contours, _ = cv2.findContours(
                encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Mặt nạ vùng đệm phương tiện giao thông di động (Vehicle Exclusion Mask)
            # Dùng để phân biệt và loại trừ hoàn toàn các vết đỏ bé li ti do xe máy/ô tô di chuyển qua
            if args.static_only:
                vehicle_ids = [
                    k for k, v in labels.items()
                    if v in {"car", "truck", "bus", "motorcycle", "bicycle", "rider", "person"}
                ]
                vehicle_mask = np.isin(pred, vehicle_ids).astype(np.uint8) * 255
                ksize = max(5, args.vehicle_buffer | 1)
                vehicle_dilated = cv2.dilate(
                    vehicle_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)),
                )
            else:
                vehicle_dilated = None

            candidate_detections = []
            encroachment_info = []
            total_encroach_area = 0
            max_encroach_depth = 0.0
            deepest_pt_yolo = None
            deepest_pt_seg = None
            encroached_side = "Trái"

            # Sắp xếp các contour theo diện tích giảm dần (ưu tiên vật thể lấn chiếm tĩnh chính)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in sorted_contours:
                area = cv2.contourArea(cnt)
                if area < args.min_encroach_area:
                    continue

                # 0. BỘ LỌC MÉP GÓC MÀN HÌNH / NẮP CAPO: Loại trừ hoàn toàn dị vật giả ở góc mép ảnh
                if is_corner_artifact(cnt, out_width, out_height, y_top, y_bottom):
                    continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                # BỘ LỌC DẢI MỎNG CHÂN TRỜI (Top Horizon Sliver Filter):
                # Các dải mỏng ngang nằm sát mép đường chân trời (y_top) là vạch kẻ xa / biển báo / cây cối
                if by <= y_top + 5 and bh <= 45:
                    continue

                cnt_mask = np.zeros((out_height, out_width), dtype=np.uint8)
                cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
                cnt_px = np.sum(cnt_mask == 1)

                # 1. BỘ LỌC VẬT THỂ TĨNH: Loại trừ đốm sinh ra do xe máy / ô tô di chuyển (bóng xe, vết cắt bánh xe...)
                if vehicle_dilated is not None:
                    overlap_vehicle = np.sum((cnt_mask == 1) & (vehicle_dilated == 255))
                    if overlap_vehicle / max(1, cnt_px) > 0.45:
                        # Hơn 45% diện tích nằm sát/dính vào phương tiện di chuyển -> Bỏ qua
                        continue

                # 2. Lọc tiếp xúc mặt đường (Road Adjacency): vật thể phải tiếp xúc/lấn vào lòng đường xe chạy
                road_contact = np.sum((cnt_mask == 1) & (road_dilated == 1))
                if road_contact < 50:
                    continue

                # 3. BỘ LỌC CÂY XANH / HẠ TẦNG DẢI PHÂN CÁCH (Greenery & Median Infrastructure Filter):
                # Cây xanh trồng trên dải phân cách (vegetation), thành bê tông dải phân cách (wall/fence),
                # cột đèn/biển báo (pole) là kiến trúc hạ tầng cố định, không phải vật thể lạ lấn chiếm.
                infra_px = np.sum(np.isin(
                    pred[cnt_mask == 1],
                    [k for k, v in labels.items() if v in {"vegetation", "wall", "fence", "pole"}]
                ))
                if infra_px / max(1, cnt_px) > 0.45:
                    continue

                if mode == "hybrid":
                    # Trong chế độ Hybrid: Loại trừ nếu phần lớn diện tích thuộc về hạ tầng cố định
                    # (vỉa hè, bóng trụ cầu metro, cột đèn, biển báo, cây xanh) theo dự đoán của YOLO Semantic.
                    yolo_infra_px = np.sum(np.isin(
                        yolo_pred[cnt_mask == 1],
                        [1, 2, 5, 7, 8, 9]  # sidewalk, building/pillar, pole, sign, vegetation, terrain
                    ))
                    if yolo_infra_px / max(1, cnt_px) > args.infra_thresh:
                        continue
                    # Riêng vỉa hè (curb/sidewalk): Nếu hơn 25% là vỉa hè theo YOLO -> Bỏ qua báo động giả
                    yolo_sw_px = np.sum(yolo_pred[cnt_mask == 1] == 1)
                    if yolo_sw_px / max(1, cnt_px) > 0.25:
                        continue

                pts = cnt.reshape(-1, 2)

                # 3. Tính khoảng cách từ đường biên cong chuẩn đến điểm lấn sâu nhất
                valid_pts = pts[pts[:, 1] < (y_bottom - 15)]
                cnt_max_depth = 0.0
                cnt_deepest_yolo = None
                cnt_deepest_seg = None
                cnt_side = "Trái"
                if len(valid_pts) > 0 and left_x_by_y is not None:
                    for pt in valid_pts:
                        py = int(np.clip(pt[1], 0, out_height - 1))
                        lx = left_x_by_y[py]
                        rx = right_x_by_y[py]
                        dl = pt[0] - lx
                        dr = rx - pt[0]
                        cur_depth = min(dl, dr)
                        if cur_depth > cnt_max_depth:
                            cnt_max_depth = cur_depth
                            edge_x = int(lx if dl < dr else rx)
                            cnt_deepest_yolo = (edge_x, py)
                            cnt_deepest_seg = (int(pt[0]), py)
                            cnt_side = "Trái" if dl < dr else "Phải"

                # BỘ LỌC ĐỘ SÂU LẤN CHIẾM TỐI THIỂU:
                # Chỉ coi là vật thể lấn chiếm nếu độ sâu ăn vào lòng đường tối thiểu args.min_encroach_depth (mặc định: 30 px)
                if cnt_max_depth < args.min_encroach_depth:
                    continue

                candidate_detections.append({
                    "cnt": cnt,
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "area": float(area),
                    "depth": float(cnt_max_depth),
                    "yolo_pt": cnt_deepest_yolo,
                    "seg_pt": cnt_deepest_seg,
                    "side": cnt_side,
                })

            # Cập nhật bộ theo dõi (Object Tracker): Chỉ xác nhận khi vật thể tồn tại ổn định >= min_duration (2.0s)
            active_tracks, finished_tracks = tracker.update(candidate_detections, frame_idx)
            is_alarm_active = len(active_tracks) > 0

            # Ghi nhận các sự việc lấn chiếm đã kết thúc vào nhật ký
            for track in finished_tracks:
                peak_f = track["peak_frame"]
                img_name = f"incident_frame_{peak_f:04d}.jpg"
                img_path = incidents_dir / img_name
                if args.save_incidents and track.get("peak_frame_rgb") is not None:
                    cv2.imwrite(
                        str(img_path),
                        cv2.cvtColor(track["peak_frame_rgb"], cv2.COLOR_RGB2BGR),
                    )
                record = {
                    "incident_id": len(incidents_log) + 1,
                    "type": "obstacle",
                    "peak_frame": peak_f,
                    "timestamp_sec": track["start_time_sec"],
                    "duration_sec": round(track["total_seen"] / out_fps, 2),
                    "duration_frames": track["total_seen"],
                    "side": encroached_side,
                    "max_area_px": int(track["max_area"]),
                    "max_depth_px": round(track["max_depth"], 1),
                    "bounding_box_xywh": list(track["bbox"]),
                    "evidence_image": img_name if args.save_incidents else None,
                }
                incidents_log.append(record)

            # Cập nhật bộ theo dõi hàng rong / buôn bán lấn chiếm mép đường
            active_vendors = []
            if vendor_tracker is not None:
                vendor_cands = vendor_tracker.find_candidates(
                    pred,
                    labels,
                    legal_corridor_poly,
                    left_x_by_y,
                    right_x_by_y,
                    args.vendor_edge_thresh,
                    out_height,
                    out_width,
                )
                active_vendors, finished_vendors = vendor_tracker.update(vendor_cands, frame_idx)

                for v_track in finished_vendors:
                    peak_f = v_track["peak_frame"]
                    img_name = f"vendor_frame_{peak_f:04d}.jpg"
                    img_path = incidents_dir / img_name
                    if args.save_incidents and v_track.get("peak_frame_rgb") is not None:
                        cv2.imwrite(
                            str(img_path),
                            cv2.cvtColor(v_track["peak_frame_rgb"], cv2.COLOR_RGB2BGR),
                        )
                    record = {
                        "incident_id": len(incidents_log) + 1,
                        "type": "street_vendor",
                        "peak_frame": peak_f,
                        "timestamp_sec": v_track["start_time_sec"],
                        "duration_sec": round(v_track["total_seen"] / out_fps, 2),
                        "duration_frames": v_track["total_seen"],
                        "side": v_track["side"],
                        "has_vehicle": v_track.get("has_vehicle", False),
                        "dist_to_edge_px": round(v_track["dist_to_edge"], 1),
                        "road_depth_px": round(v_track.get("road_depth", 0), 1),
                        "bounding_box_xywh": list(v_track["bbox"]),
                        "evidence_image": img_name if args.save_incidents else None,
                    }
                    incidents_log.append(record)

            if active_tracks:
                total_encroach_area = sum(t["area"] for t in active_tracks)
                max_encroach_depth = max(t["depth"] for t in active_tracks)
                deepest_pt_yolo = active_tracks[0]["yolo_pt"]
                deepest_pt_seg = active_tracks[0]["seg_pt"]
                cur_w = (
                    (right_x_by_y[y_bottom] - left_x_by_y[y_bottom])
                    if (right_x_by_y is not None and left_x_by_y is not None)
                    else max(1.0, out_width * 0.8)
                )
                encroach_pct = (max_encroach_depth / max(1.0, cur_w)) * 100
            else:
                total_encroach_area = 0
                max_encroach_depth = 0.0
                deepest_pt_yolo = None
                deepest_pt_seg = None
                encroach_pct = 0.0

            # 5. Vẽ trực quan Video Overlay (overlay.mp4)
            overlay = frame_rgb.copy()

            # Phủ màu nền SegFormer: Mặt đường = Xanh lá (0, 255, 0), Vỉa hè = Cam (255, 165, 0)
            if args.alpha > 0.0:
                mask_has_seg = mask_road | mask_sw
                if np.any(mask_has_seg):
                    seg_color = np.zeros_like(overlay)
                    seg_color[mask_road] = COLORS["road"]          # Xanh lá: mặt đường xe chạy
                    seg_color[mask_sw] = COLORS["sidewalk"]        # Cam: vỉa hè / lề đường
                    blended = cv2.addWeighted(overlay, 1.0 - args.alpha, seg_color, args.alpha, 0)
                    overlay[mask_has_seg] = blended[mask_has_seg]

            # Vẽ vùng vật thể lấn chiếm (VIỀN ĐỎ + TÔ ĐỎ CAM) khi cảnh báo được xác nhận (tracker confirmed >= 2s)
            if is_alarm_active and active_tracks:
                fill_layer = np.zeros_like(overlay)
                for track in active_tracks:
                    obs_mask = get_solid_obstacle_mask(
                        track, out_height, out_width, left_x_by_y, right_x_by_y
                    )
                    obs_cnts, _ = cv2.findContours(
                        obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if obs_cnts:
                        # Viền cong màu đỏ đậm dày 4px bao quanh toàn bộ vật thể
                        cv2.drawContours(overlay, obs_cnts, -1, (255, 0, 0), 4, cv2.LINE_AA)
                    # Tô màu đỏ cam bán trong suốt toàn bộ diện tích lấn chiếm đặc kín
                    fill_layer[obs_mask == 255] = (255, 50, 0)

                cv2.addWeighted(overlay, 1.0, fill_layer, 0.45, 0, dst=overlay)

                # Vẽ mũi tên đo khoảng cách từ đường biên chuẩn thẳng (YOLO) vào điểm lấn chiếm sâu nhất
                if deepest_pt_yolo is not None and deepest_pt_seg is not None and max_encroach_depth > 5:
                    cv2.arrowedLine(overlay, deepest_pt_yolo, deepest_pt_seg, (255, 0, 0), 3, tipLength=0.2)
                    cv2.circle(overlay, deepest_pt_seg, 6, (255, 255, 0), -1)
                    cv2.putText(
                        overlay,
                        f"Lan chiem: {int(max_encroach_depth)} px",
                        (deepest_pt_seg[0] + 15, max(30, deepest_pt_seg[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Lưu ảnh đỉnh điểm cho các track đang theo dõi
                for track in active_tracks:
                    if track["peak_frame"] == frame_idx:
                        track["peak_frame_rgb"] = overlay.copy()

            # Vẽ cảnh báo người bán hàng rong lấn chiếm (VIỀN TÍM / HỒNG ĐẬM + BBOX + MŨI TÊN)
            if active_vendors:
                vendor_layer = np.zeros_like(overlay)
                for v in active_vendors:
                    vx, vy, vw, vh = v["bbox"]
                    cnt = v.get("cnt")
                    v_sec = v["total_seen"] / out_fps
                    color_vendor = (255, 0, 255)  # Magenta / Tím đậm nổi bật

                    if cnt is not None:
                        cv2.drawContours(vendor_layer, [cnt], -1, color_vendor, -1)
                        cv2.drawContours(overlay, [cnt], -1, color_vendor, 3, cv2.LINE_AA)

                    cv2.rectangle(overlay, (vx, vy), (vx + vw, vy + vh), color_vendor, 2, cv2.LINE_AA)
                    label = f"HANG RONG #{v['id']} ({v_sec:.1f}s)"
                    if v.get("has_vehicle"):
                        label += " [XE]"

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(overlay, (vx, max(0, vy - 24)), (vx + tw + 10, vy), (40, 0, 40), -1)
                    cv2.putText(
                        overlay,
                        label,
                        (vx + 5, max(16, vy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Kẻ mũi tên từ vị trí vendor sang mép lòng đường
                    cx, cy = v["centroid"]
                    ex = v.get("edge_x", cx)
                    cv2.line(overlay, (cx, cy), (ex, cy), (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(overlay, (ex, cy), 4, (255, 255, 0), -1)

                    if v.get("peak_frame_rgb") is None or (v["total_seen"] % 15 == 0):
                        v["peak_frame"] = frame_idx
                        v["peak_frame_rgb"] = overlay.copy()

                cv2.addWeighted(overlay, 1.0, vendor_layer, 0.40, 0, dst=overlay)

            # Vẽ 2 đường biên mép đường nếu được bật (mặc định: ẩn theo yêu cầu)
            if args.show_boundary_lines:
                if pts_left_curve is not None and len(pts_left_curve) > 1:
                    cv2.polylines(overlay, [pts_left_curve], False, line_color_rgb, 3, cv2.LINE_AA)
                if pts_right_curve is not None and len(pts_right_curve) > 1:
                    cv2.polylines(overlay, [pts_right_curve], False, line_color_rgb, 3, cv2.LINE_AA)

            # Bảng HUD hiển thị thông số và cảnh báo góc trên
            cv2.rectangle(overlay, (15, 15), (660, 105), (10, 10, 15), -1)
            cv2.rectangle(overlay, (15, 15), (660, 105), (80, 80, 100), 1)

            if is_alarm_active and active_tracks:
                max_track_sec = max(t["total_seen"] / out_fps for t in active_tracks)
                cv2.putText(
                    overlay,
                    f"[CANH BAO] PHAT HIEN LAN CHIEM BIEN {encroached_side.upper()}!",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"Dien tich: {total_encroach_area:.0f} px | Do sau: {max_encroach_depth:.1f} px | Time: {max_track_sec:.1f}s",
                    (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            elif active_vendors:
                v_side = active_vendors[0]["side"]
                max_v_sec = max(v["total_seen"] / out_fps for v in active_vendors)
                cv2.putText(
                    overlay,
                    f"[CANH BAO] PHAT HIEN HANG RONG LAN CHIEM BIEN {v_side.upper()}!",
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"So luong: {len(active_vendors)} | Thoi gian: {max_v_sec:.1f}s | Sat mep long duong",
                    (25, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    overlay,
                    "[TRANG THAI] LONG DUONG THONG THOANG",
                    (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    "Khong phat hien vat the / hang rong lan chiem",
                    (25, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

            current_fps = (processed_count + 1) / max(0.001, (time.perf_counter() - start_time))
            cv2.putText(
                overlay,
                f"Che do: {mode.upper()} | Frame: {frame_idx}/{total_frames} | Toc do: {current_fps:.1f} FPS",
                (25, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Ghi video overlay
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            overlay_writer.write(overlay_bgr)

            # 6. Ghi video Mask (road_mask.mp4 và encroach_mask.mp4)
            if args.save_masks:
                # Video road_mask.mp4: Hành lang lòng đường chuẩn (road + xe = TRẮNG 255, nền = ĐEN 0)
                # Mặt đường và xe cộ được gộp làm 1, không còn lỗ thủng xe màu đen
                road_mask_disp = legal_corridor_poly.copy()
                road_bgr = cv2.cvtColor(road_mask_disp, cv2.COLOR_GRAY2BGR)
                road_writer.write(road_bgr)

                # Video encroach_mask.mp4:
                # Toàn bộ hành lang lòng đường là màu trắng (255), vật thể lấn chiếm được TÔ ĐEN ĐẶC (0) hoàn toàn
                mask_encroach_disp = legal_corridor_poly.copy()
                if is_alarm_active and active_tracks:
                    for track in active_tracks:
                        obs_mask = get_solid_obstacle_mask(
                            track, out_height, out_width, left_x_by_y, right_x_by_y
                        )
                        mask_encroach_disp[obs_mask == 255] = 0
                if active_vendors:
                    v_cnts = [v["cnt"] for v in active_vendors if v.get("cnt") is not None]
                    if v_cnts:
                        cv2.drawContours(mask_encroach_disp, v_cnts, -1, 0, -1)

                encroach_bgr = cv2.cvtColor(mask_encroach_disp, cv2.COLOR_GRAY2BGR)
                encroach_writer.write(encroach_bgr)

            processed_count += 1
            if processed_count % 30 == 0:
                elapsed = time.perf_counter() - start_time
                avg_fps = processed_count / max(elapsed, 1e-6)
                progress_pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"[INFO] Frame {frame_idx}/{total_frames} ({progress_pct:.1f}%) | "
                    f"Đã xử lý: {processed_count} frames | Tốc độ: {avg_fps:.2f} FPS"
                )

    finally:
        cap.release()
        overlay_writer.release()
        if road_writer is not None:
            road_writer.release()
        if encroach_writer is not None:
            encroach_writer.release()

    elapsed = time.perf_counter() - start_time

    # Lưu các sự việc đã xác nhận còn đang theo dõi ở frame cuối (nếu có)
    for track in tracker.tracks.values():
        if track["confirmed"] and not track["saved"]:
            peak_f = track["peak_frame"]
            img_name = f"incident_frame_{peak_f:04d}.jpg"
            img_path = incidents_dir / img_name
            if args.save_incidents and track.get("peak_frame_rgb") is not None:
                cv2.imwrite(
                    str(img_path),
                    cv2.cvtColor(track["peak_frame_rgb"], cv2.COLOR_RGB2BGR),
                )
            record = {
                "incident_id": len(incidents_log) + 1,
                "type": "obstacle",
                "peak_frame": peak_f,
                "timestamp_sec": track["start_time_sec"],
                "duration_sec": round(track["total_seen"] / out_fps, 2),
                "duration_frames": track["total_seen"],
                "side": track.get("side", encroached_side),
                "max_area_px": int(track["max_area"]),
                "max_depth_px": round(track["max_depth"], 1),
                "bounding_box_xywh": list(track["bbox"]),
                "evidence_image": img_name if args.save_incidents else None,
            }
            incidents_log.append(record)
            track["saved"] = True

    # Lưu các sự việc hàng rong lấn chiếm đã xác nhận còn đang theo dõi ở frame cuối
    if vendor_tracker is not None:
        for track in vendor_tracker.tracks.values():
            if track["confirmed"] and not track["saved"]:
                peak_f = track["peak_frame"]
                img_name = f"vendor_frame_{peak_f:04d}.jpg"
                img_path = incidents_dir / img_name
                if args.save_incidents and track.get("peak_frame_rgb") is not None:
                    cv2.imwrite(
                        str(img_path),
                        cv2.cvtColor(track["peak_frame_rgb"], cv2.COLOR_RGB2BGR),
                    )
                record = {
                    "incident_id": len(incidents_log) + 1,
                    "type": "street_vendor",
                    "peak_frame": peak_f,
                    "timestamp_sec": track["start_time_sec"],
                    "duration_sec": round(track["total_seen"] / out_fps, 2),
                    "duration_frames": track["total_seen"],
                    "side": track["side"],
                    "has_vehicle": track.get("has_vehicle", False),
                    "dist_to_edge_px": round(track["dist_to_edge"], 1),
                    "road_depth_px": round(track.get("road_depth", 0), 1),
                    "bounding_box_xywh": list(track["bbox"]),
                    "evidence_image": img_name if args.save_incidents else None,
                }
                incidents_log.append(record)
                track["saved"] = True

    # Xuất file log JSON cho trung tâm kiểm duyệt
    if args.save_incidents:
        json_path = incidents_dir / "incidents_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(incidents_log, f, indent=2, ensure_ascii=False)

    print("\n[HOÀN THÀNH]")
    print(f"- Số frame đã xử lý: {processed_count}/{total_frames}")
    print(f"- Thời gian thực thi: {elapsed:.2f}s ({processed_count / max(elapsed, 1e-6):.2f} FPS)")
    print(f"- Video overlay trực quan (viền đỏ đống cát): {overlay_path}")
    if args.save_masks:
        print(f"- Video mặt đường segment: {road_path}")
        print(f"- Video lấn chiếm (hành lang = trắng, đống cát = đen): {encroach_path}")
    if args.save_incidents:
        print(f"- Thư mục bằng chứng vi phạm: {incidents_dir}")
        print(f"- File nhật ký sự việc JSON: {json_path}")
        print(f"- Tổng số sự cố lấn chiếm được ghi nhận: {len(incidents_log)}")


if __name__ == "__main__":
    main()

