import cv2
import numpy as np
import torch
from pathlib import Path
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

labels = seg_model.config.id2label
road_id = 0
sidewalk_id = 1

cap = cv2.VideoCapture("video/test2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

y_top = int(h * 0.35)
y_bottom = int(h * 0.98)

prev_probs = None
prev_curve_left = None
prev_curve_right = None
prev_yolo_l = None
prev_yolo_r = None

for frame_idx in range(145):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    if frame_idx < 70:
        continue
        
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # YOLO
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred_raw = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred_raw.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        yolo_pred = yolo_pred_raw

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

    mask_sw = (pred == sidewalk_id) | (yolo_pred == sidewalk_id)
    mask_road = (pred == road_id) & (~mask_sw)
    veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
    veh_dil = cv2.dilate(veh_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    (la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(
        yolo_pred, h, w, road_id=road_id, sidewalk_id=sidewalk_id,
        prev_left=prev_yolo_l, prev_right=prev_yolo_r, alpha=0.15
    )
    prev_yolo_l, prev_yolo_r = (la, lb), (ra, rb)

    cnts, _ = cv2.findContours(mask_road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_road = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        max_area = max(cv2.contourArea(c) for c in cnts)
        for c in cnts:
            if cv2.contourArea(c) > max(1000.0, max_area * 0.05):
                cv2.drawContours(clean_road, [c], -1, 255, -1)

    all_ys = np.arange(y_top, y_bottom + 1)
    left_x = np.full(h, -1.0, dtype=np.float32)
    right_x = np.full(h, -1.0, dtype=np.float32)

    top_anchor_ys = np.arange(y_top, min(y_top + 25, y_bottom))
    for ty in top_anchor_ys:
        left_x[ty] = np.clip(la * ty + lb, 0, w * 0.5)
        right_x[ty] = np.clip(ra * ty + rb, w * 0.5, w - 1)

    for y in all_ys:
        if y in top_anchor_ys:
            continue
        r_xs = np.where(clean_road[y] > 0)[0]
        v_xs = np.where(veh_dil[y] > 0)[0]
        exp_lx = la * y + lb
        exp_rx = ra * y + rb

        if len(r_xs) > 0:
            cur_lx = float(r_xs[0])
            is_occluded = False
            if len(v_xs) > 0 and (v_xs[0] <= cur_lx + 35):
                is_occluded = True
            if cur_lx > exp_lx + 45:
                is_occluded = True
            if not is_occluded:
                left_x[y] = cur_lx
            else:
                sw_xs = np.where((mask_sw[y] > 0) & (veh_dil[y] == 0))[0]
                if len(sw_xs) > 0 and abs(sw_xs[-1] - exp_lx) < 60:
                    left_x[y] = float(sw_xs[-1])

        if len(r_xs) > 0:
            cur_rx = float(r_xs[-1])
            is_occluded = False
            if len(v_xs) > 0 and (v_xs[-1] >= cur_rx - 35):
                is_occluded = True
            if cur_rx < exp_rx - 45:
                is_occluded = True
            if not is_occluded:
                right_x[y] = cur_rx
            else:
                sw_xs = np.where((mask_sw[y] > 0) & (veh_dil[y] == 0))[0]
                sw_right = sw_xs[sw_xs > w * 0.5] if len(sw_xs) else []
                if len(sw_right) > 0 and abs(sw_right[0] - exp_rx) < 60:
                    right_x[y] = float(sw_right[0])

    valid_l = np.where(left_x >= 0)[0]
    valid_l = valid_l[(valid_l >= y_top) & (valid_l <= y_bottom)]
    interp_l = np.interp(all_ys, valid_l, left_x[valid_l]) if len(valid_l) >= 2 else np.clip(la * all_ys + lb, 0, w - 1)

    valid_r = np.where(right_x >= 0)[0]
    valid_r = valid_r[(valid_r >= y_top) & (valid_r <= y_bottom)]
    interp_r = np.interp(all_ys, valid_r, right_x[valid_r]) if len(valid_r) >= 2 else np.clip(ra * all_ys + rb, 0, w - 1)

    ksize = 61
    sigma = 18.0
    kg = cv2.getGaussianKernel(ksize, sigma).flatten()
    half = ksize // 2
    pad_l = np.pad(interp_l, half, mode="edge")
    pad_r = np.pad(interp_r, half, mode="edge")
    smooth_l = np.convolve(pad_l, kg, mode="valid")
    smooth_r = np.convolve(pad_r, kg, mode="valid")

    if prev_curve_left is not None and len(prev_curve_left) == len(smooth_l):
        smooth_l = 0.20 * smooth_l + 0.80 * prev_curve_left
    if prev_curve_right is not None and len(prev_curve_right) == len(smooth_r):
        smooth_r = 0.20 * smooth_r + 0.80 * prev_curve_right
    prev_curve_left = smooth_l
    prev_curve_right = smooth_r

    smooth_l = np.minimum(smooth_l, smooth_r - 20.0)

    pts_l = np.column_stack([np.clip(smooth_l, 0, w - 1).astype(np.int32), all_ys])
    pts_r = np.column_stack([np.clip(smooth_r, 0, w - 1).astype(np.int32), all_ys])
    poly = np.vstack([pts_l, pts_r[::-1]])
    legal_corridor_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(legal_corridor_poly, [poly], 255)

    left_x_by_y = np.full(h, -1.0, dtype=np.float32)
    right_x_by_y = np.full(h, -1.0, dtype=np.float32)
    left_x_by_y[all_ys] = smooth_l
    right_x_by_y[all_ys] = smooth_r

    normal_mask = np.isin(pred, [road_id, sidewalk_id] + [k for k, v in labels.items() if v in rve.CORRIDOR_CLASSES]).astype(np.uint8) * 255
    normal_mask = cv2.morphologyEx(normal_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    road_dilated = cv2.dilate((pred == road_id).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    encroach_raw = (legal_corridor_poly == 255) & (normal_mask == 0) & (road_dilated > 0)
    encroach_mask = cv2.morphologyEx(encroach_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_ids = [k for k, v in labels.items() if v in {"car", "truck", "bus", "motorcycle", "bicycle", "rider", "person"}]
    vehicle_mask = np.isin(pred, vehicle_ids).astype(np.uint8) * 255
    vehicle_dilated = cv2.dilate(vehicle_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))

    dets = []
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
        if len(valid_pts) > 0:
            for pt in valid_pts:
                py = int(np.clip(pt[1], 0, h - 1))
                lx = left_x_by_y[py]
                rx = right_x_by_y[py]
                dl = pt[0] - lx
                dr = rx - pt[0]
                cnt_max_depth = max(cnt_max_depth, min(dl, dr))

        dyn_min_depth = max(25.0, 35.0 * (0.7 + 0.3 * rel_y))
        if cnt_max_depth < dyn_min_depth:
            continue

        dets.append((bx, by, bw, bh, area, cnt_max_depth))

    print(f"Frame {frame_idx:3d}: {len(dets)} dets -> {[(d[0], d[1], d[2], d[3], int(d[4])) for d in dets]}")
