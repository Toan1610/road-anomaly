import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

from run_video_encroachment import (
    CITYSCAPES_19_LABELS,
    CORRIDOR_CLASSES,
    EncroachTracker,
    is_corner_artifact,
)
from run_video_encroachment_mask2former import (
    infer_mask2former_probs,
    GpuMorphology,
    make_class_lut,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
road_id = 0
sidewalk_id = 1

# Infrastructure: sidewalk (1), fence (4), pole (5), vegetation (8), terrain (9)
LUT_INFRA = make_class_lut([1, 4, 8, 9])
LUT_INFRA_CROP2 = make_class_lut([1, 4, 5, 8])

# Moving vehicles:
LUT_VEHICLE = make_class_lut([11, 12, 13, 14, 15, 16, 17])
normal_classes = [road_id, sidewalk_id] + [k for k, v in labels.items() if v in CORRIDOR_CLASSES]
LUT_NORMAL = make_class_lut(normal_classes)

gpu_morph = GpuMorphology(device)
tracker = EncroachTracker(fps=fps, min_duration_sec=0.8)

prev_curve_left = None
prev_curve_right = None

y_top = int(H * 0.35)
y_bottom = int(H * 0.95)

def extract_corridor_fixed(mask_road, mask_sw, veh_mask, h, w, y_top, y_bot, prev_left, prev_right, alpha=0.15, mask_infra=None):
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

    veh_dil = cv2.dilate(veh_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    la = (0.25 * w - 0.08 * w) / (y_top - y_bot)
    lb = 0.08 * w - la * y_bot
    ra = (0.75 * w - 0.92 * w) / (y_top - y_bot)
    rb = 0.92 * w - ra * y_bot

    top_anchor_ys = np.arange(y_top, min(y_top + 25, y_bot))
    for ty in top_anchor_ys:
        left_x[ty] = np.clip(la * ty + lb, 0, w * 0.5)
        right_x[ty] = np.clip(ra * ty + rb, w * 0.5, w - 1)

    clean_bool = clean_road > 0
    veh_bool = veh_dil > 0
    check_infra = mask_infra if mask_infra is not None else mask_sw
    infra_bool = (check_infra > 0) & (~veh_bool)

    r_has = clean_bool.any(axis=1)
    r_first = np.where(r_has, np.argmax(clean_bool, axis=1), -1)
    r_last = np.where(r_has, w - 1 - np.argmax(clean_bool[:, ::-1], axis=1), -1)

    v_has = veh_bool.any(axis=1)
    v_f = np.where(v_has, np.argmax(veh_bool, axis=1), -1)
    v_l = np.where(v_has, w - 1 - np.argmax(veh_bool[:, ::-1], axis=1), -1)

    for y in all_ys:
        if y in top_anchor_ys or not r_has[y]:
            continue
        exp_lx = la * y + lb
        exp_rx = ra * y + rb
        cur_lx = float(r_first[y])
        cur_rx = float(r_last[y])

        # Left side occlusion (vehicle or large indentation)
        is_occluded_l = (v_has[y] and v_f[y] <= cur_lx + 35) or (cur_lx > exp_lx + 45)
        if not is_occluded_l:
            left_x[y] = cur_lx

        # Right side: follow road edge faithfully unless occluded by moving vehicle
        is_occluded_r = (v_has[y] and v_l[y] >= cur_rx - 35)
        if not is_occluded_r:
            right_x[y] = cur_rx

    valid_l = np.where(left_x >= 0)[0]
    valid_l = valid_l[(valid_l >= y_top) & (valid_l <= y_bot)]
    interp_left = np.interp(all_ys, valid_l, left_x[valid_l]) if len(valid_l) >= 2 else np.clip(la * all_ys + lb, 0, w - 1)

    valid_r = np.where(right_x >= 0)[0]
    valid_r = valid_r[(valid_r >= y_top) & (valid_r <= y_bot)]
    interp_right = np.interp(all_ys, valid_r, right_x[valid_r]) if len(valid_r) >= 2 else np.clip(ra * all_ys + rb, 0, w - 1)

    # Bridge across foreign obstacle (sand pile) on left side:
    for idx, y in enumerate(all_ys):
        yl = la * y + lb
        cl = interp_left[idx]
        if yl < cl - 20:
            x1 = max(0, int(yl))
            x2 = min(w, int(cl))
            if x2 > x1 and np.mean(check_infra[y, x1:x2]) < 0.35:
                interp_left[idx] = yl

    # Smooth
    ksize, sigma = 61, 18.0
    kg = cv2.getGaussianKernel(ksize, sigma).flatten()
    half = ksize // 2
    pad_l = np.pad(interp_left, half, mode="edge")
    pad_r = np.pad(interp_right, half, mode="edge")
    smooth_l = np.convolve(pad_l, kg, mode="valid")
    smooth_r = np.convolve(pad_r, kg, mode="valid")

    if prev_left is not None and len(prev_left) == len(smooth_l):
        smooth_l = alpha * smooth_l + (1.0 - alpha) * prev_left
    if prev_right is not None and len(prev_right) == len(smooth_r):
        smooth_r = alpha * smooth_r + (1.0 - alpha) * prev_right

    smooth_l = np.minimum(smooth_l, smooth_r - 20.0)
    pts_l = np.column_stack([np.clip(smooth_l, 0, w - 1).astype(np.int32), all_ys])
    pts_r = np.column_stack([np.clip(smooth_r, 0, w - 1).astype(np.int32), all_ys])
    poly = np.vstack([pts_l, pts_r[::-1]])

    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 255)

    lx_by_y = np.full(h, np.nan, dtype=np.float32)
    rx_by_y = np.full(h, np.nan, dtype=np.float32)
    lx_by_y[all_ys] = smooth_l
    rx_by_y[all_ys] = smooth_r

    return poly_mask, pts_l, pts_r, lx_by_y, rx_by_y, smooth_l, smooth_r

frame_idx = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret: break
    frame_idx += 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    probs = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, True)
    pred = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    mask_sw = (pred == sidewalk_id)
    mask_infra = LUT_INFRA[pred].astype(np.uint8)
    mask_road = (pred == road_id) & (~mask_sw)
    veh_mask = LUT_VEHICLE[pred] & (~mask_sw)
    
    (
        legal_corridor_poly,
        pts_left_curve,
        pts_right_curve,
        left_x_by_y,
        right_x_by_y,
        prev_curve_left,
        prev_curve_right,
    ) = extract_corridor_fixed(
        mask_road, mask_sw, veh_mask, H, W, y_top, y_bottom,
        prev_curve_left, prev_curve_right, alpha=0.15, mask_infra=mask_infra
    )
    
    normal_mask = (LUT_NORMAL[pred]).astype(np.uint8) * 255
    normal_mask = gpu_morph.close(normal_mask, iterations=7)
    
    horizon_cutoff = max(int(H * 0.50), y_top + 65)
    is_road_level = (np.arange(H)[:, None] >= horizon_cutoff)
    
    # Anomaly core: everything inside legal corridor that is not drivable surface and not moving vehicle
    # Connected to the road: we check contour road contact
    anomaly_core = (
        (legal_corridor_poly == 255)
        & (normal_mask == 0)
        & is_road_level
        & (~veh_mask)
    )
    
    obs_closed = gpu_morph.close(anomaly_core.astype(np.uint8) * 255, iterations=4)
    encroach_mask = gpu_morph.open(obs_closed, iterations=3)
    
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_dilated = gpu_morph.dilate((pred == road_id).astype(np.uint8), iterations=35)
    
    candidate_detections = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0: continue
        
        rel_y = max(0.0, min(1.0, (by + bh * 0.5 - y_top) / max(1.0, y_bottom - y_top)))
        if area < max(400.0, 600.0 * (0.8 + 0.4 * rel_y)): continue
        if is_corner_artifact(cnt, W, H, y_top, y_bottom): continue
        if by <= y_top + 5 and bh <= 45: continue
        
        cnt_mask_crop = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(cnt_mask_crop, [cnt - np.array([bx, by])], -1, 1, -1)
        cnt_px = int(np.count_nonzero(cnt_mask_crop))
        if cnt_px <= 0: continue
        
        road_crop = road_dilated[by : by + bh, bx : bx + bw]
        if int(np.count_nonzero((cnt_mask_crop == 1) & (road_crop == 1))) < 30: continue
        
        pred_crop = pred[by : by + bh, bx : bx + bw]
        if int(np.sum(LUT_INFRA_CROP2[pred_crop[cnt_mask_crop == 1]])) / cnt_px > 0.40: continue
        
        cnt_side = "Trái" if (bx + bw * 0.35) < (W * 0.5) else "Phải"
        mid_y = int(np.clip(by + bh * 0.5, 0, H - 1))
        
        if cnt_side == "Trái":
            curb_x = left_x_by_y[mid_y] if left_x_by_y is not None else 0
            if (bx - curb_x) > 45: continue
        else:
            curb_x = right_x_by_y[mid_y] if right_x_by_y is not None else (W - 1)
            if (curb_x - (bx + bw)) > 45: continue
        
        pts = cnt.reshape(-1, 2)
        valid_pts = pts[(pts[:, 1] >= y_top) & (pts[:, 1] < int(H * 0.90))]
        cnt_max_depth = 0.0
        cnt_deepest_pt = None
        if len(valid_pts) > 0 and left_x_by_y is not None and right_x_by_y is not None:
            for pt in valid_pts:
                py = int(np.clip(pt[1], 0, H - 1))
                lx = np.clip(left_x_by_y[py], 0.0, W - 1.0)
                rx = np.clip(right_x_by_y[py], 0.0, W - 1.0)
                depth = float(pt[0] - lx) if cnt_side == "Trái" else float(rx - pt[0])
                if depth > cnt_max_depth:
                    cnt_max_depth = depth
                    cnt_deepest_pt = (int(pt[0]), int(pt[1]))
        
        if cnt_max_depth < 40.0: continue
        candidate_detections.append({
            "cnt": cnt, "area": area, "depth": cnt_max_depth,
            "side": cnt_side, "bbox": (bx, by, bw, bh),
            "yolo_pt": cnt_deepest_pt, "seg_pt": cnt_deepest_pt,
        })
    
    active_confirmed, finished_tracks = tracker.update(candidate_detections, frame_idx)

confirmed = tracker.flush_confirmed()
print("\n================ FINAL CONFIRMED INCIDENTS ================")
print(f"Total confirmed incidents: {len(confirmed)}")
for inc in confirmed:
    print(f"Incident {inc['id']}: peak_frame={inc['peak_frame']}, side={inc['side']}, duration_frames={inc['total_seen']}, max_area={inc['max_area']}, max_depth={inc['max_depth']:.1f}")

