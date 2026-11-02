import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO
import sys
sys.path.insert(0, ".")
import run_video_encroachment as rve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()
yolo_model = YOLO("yolo26n-sem.pt")

cap = cv2.VideoCapture("video/test2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Run through frames 0 to 145 and inspect frame 98 and frame 140
tracker = rve.EncroachTracker(fps=fps, min_duration_sec=0.4, max_lost_sec=2.0)
prev_probs = None
prev_curve_left = None
prev_curve_right = None
prev_yolo_l = None
prev_yolo_r = None

for frame_idx in range(145):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # YOLO
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

    # SegFormer
    inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = seg_model(**inputs)
        logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits[0], dim=0)
        if prev_probs is not None:
            probs = 0.7 * probs + 0.3 * prev_probs
        prev_probs = probs.detach()
        pred = probs.argmax(dim=0).cpu().numpy()

    mask_sw = (pred == 1) | (yolo_pred == 1)
    mask_road = (pred == 0) & (~mask_sw)
    veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)

    (la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w, prev_left=prev_yolo_l, prev_right=prev_yolo_r)
    prev_yolo_l, prev_yolo_r = (la, lb), (ra, rb)

    corridor, pts_l, pts_r, lx_by_y, rx_by_y, smooth_l, smooth_r = rve.extract_curved_road_corridor(
        mask_road, mask_sw, veh_mask, h, w, prev_left_curve=prev_curve_left, prev_right_curve=prev_curve_right, yolo_lines=((la, lb), (ra, rb))
    )
    prev_curve_left = smooth_l
    prev_curve_right = smooth_r

    normal_mask = np.isin(pred, [0, 1] + [k for k, v in seg_model.config.id2label.items() if v in rve.CORRIDOR_CLASSES]).astype(np.uint8) * 255
    normal_mask = cv2.morphologyEx(normal_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    road_dilated = cv2.dilate((pred == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    encroach_raw = (corridor == 255) & (normal_mask == 0) & (road_dilated > 0)
    encroach_mask = cv2.morphologyEx(encroach_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_ids = [k for k, v in seg_model.config.id2label.items() if v in {"car", "truck", "bus", "motorcycle", "bicycle", "rider", "person"}]
    vehicle_dilated = cv2.dilate(np.isin(pred, vehicle_ids).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))

    candidate_dets = []
    y_top = int(h * 0.35)
    y_bottom = int(h * 0.98)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        rel_y = max(0.0, min(1.0, (by + bh * 0.5 - y_top) / max(1.0, y_bottom - y_top)))
        dyn_min_area = max(280.0, 450 * (0.6 + 0.4 * rel_y))
        if area < dyn_min_area:
            continue
        if rve.is_corner_artifact(cnt, w, h, y_top, y_bottom):
            continue
        if by <= y_top + 5 and bh <= 35:
            continue
        cnt_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
        cnt_px = np.sum(cnt_mask == 1)

        overlap_v = np.sum((cnt_mask == 1) & (vehicle_dilated == 255))
        if overlap_v / max(1, cnt_px) > 0.45:
            continue
        road_contact = np.sum((cnt_mask == 1) & (road_dilated == 1))
        if road_contact < 25:
            continue
        yolo_sw_px = np.sum(yolo_pred[cnt_mask == 1] == 1)
        if yolo_sw_px / max(1, cnt_px) > 0.35:
            continue

        pts = cnt.reshape(-1, 2)
        valid_pts = pts[pts[:, 1] < (y_bottom - 15)]
        cnt_max_depth = 0.0
        cnt_deepest_yolo = None
        cnt_deepest_seg = None
        cnt_side = "Trái"
        if len(valid_pts) > 0 and lx_by_y is not None:
            for pt in valid_pts:
                py = int(np.clip(pt[1], 0, h - 1))
                lx = lx_by_y[py]
                rx = rx_by_y[py]
                dl = pt[0] - lx
                dr = rx - pt[0]
                cur_depth = min(dl, dr)
                if cur_depth > cnt_max_depth:
                    cnt_max_depth = cur_depth
                    edge_x = int(lx if dl < dr else rx)
                    cnt_deepest_yolo = (edge_x, py)
                    cnt_deepest_seg = (int(pt[0]), py)
                    cnt_side = "Trái" if dl < dr else "Phải"

        dyn_min_depth = max(25.0, 35.0 * (0.7 + 0.3 * rel_y))
        if cnt_max_depth < dyn_min_depth:
            continue

        candidate_dets.append({
            "cnt": cnt,
            "bbox": (bx, by, bw, bh),
            "area": area,
            "depth": cnt_max_depth,
            "yolo_pt": cnt_deepest_yolo,
            "seg_pt": cnt_deepest_seg,
            "side": cnt_side,
        })

    active, finished = tracker.update(candidate_dets, frame_idx)
    if frame_idx in (98, 140):
        print(f"\n================ FRAME {frame_idx} ================")
        print(f"Active tracks: {len(active)}, Candidate dets: {len(candidate_dets)}")
        for t in active:
            print(f"Track ID {t['id']}: bbox={t['bbox']}, area={t['area']}, depth={t['depth']}, side={t['side']}, yolo_pt={t['yolo_pt']}, seg_pt={t['seg_pt']}")
            # Check obs_mask
            obs_mask = rve.get_solid_obstacle_mask(t, h, w, lx_by_y, rx_by_y)
            obs_cnts, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for oc in obs_cnts:
                obx, oby, obw, obh = cv2.boundingRect(oc)
                print(f"  obs_mask contour bbox: ({obx}, {oby}, {obw}, {obh}), area={cv2.contourArea(oc)}")
            # Where is t['cnt']?
            pts = t['cnt'].reshape(-1, 2)
            print(f"  t['cnt'] x range: [{pts[:,0].min()}, {pts[:,0].max()}], y range: [{pts[:,1].min()}, {pts[:,1].max()}]")
            for py in [oby, oby + obh//2, oby + obh - 1]:
                if 0 <= py < h:
                    print(f"  at y={py}: lx_by_y={lx_by_y[py]:.1f}")
cap.release()
