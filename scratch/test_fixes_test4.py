import cv2
import numpy as np
import torch
import sys, os

PIDNET_ROOT = os.path.abspath("PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)
    sys.path.insert(0, os.path.join(PIDNET_ROOT, "tools"))
sys.path.insert(0, os.path.abspath("."))

from ultralytics import YOLO
import models as pidnet_models

from run_video_encroachment import (
    load_pidnet_model,
    fit_dynamic_road_lines_yolo,
    EncroachTracker,
    CORRIDOR_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("Road_Segmentation_Model/yolo26n-sem.pt")
pidnet_model = load_pidnet_model("pidnet-s", "PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt", device, True)

cap = cv2.VideoCapture("video/test4.mp4")

def extract_curved_corridor_clamped(
    mask_road, mask_sw, veh_mask, h, w, y_top, y_bot,
    prev_left_curve=None, prev_right_curve=None, alpha=0.15, yolo_lines=None
):
    all_ys = np.arange(y_top, y_bot + 1)
    clean_road = cv2.morphologyEx(
        (mask_road > 0).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )
    veh_dil = cv2.dilate(
        (veh_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    left_x = np.full(h, -1.0, dtype=np.float32)
    right_x = np.full(h, -1.0, dtype=np.float32)

    if yolo_lines is not None:
        (la, lb), (ra, rb) = yolo_lines
    else:
        la = (0.25 * w - 0.08 * w) / (y_top - y_bot)
        lb = 0.08 * w - la * y_bot
        ra = (0.75 * w - 0.92 * w) / (y_top - y_bot)
        rb = 0.92 * w - ra * y_bot

    top_anchor_ys = np.arange(y_top, min(y_top + 25, y_bot))
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

        # BIEN TRAI
        if len(r_xs) > 0:
            cur_lx = float(r_xs[0])
            is_occluded = False
            if len(v_xs) > 0 and (v_xs[0] <= cur_lx + 35):
                is_occluded = True
            if cur_lx > exp_lx + 45:
                is_occluded = True
            if cur_lx < exp_lx - 50:
                cur_lx = max(cur_lx, exp_lx - 40)
            if not is_occluded:
                left_x[y] = cur_lx
            else:
                sw_xs = np.where((mask_sw[y] > 0) & (veh_dil[y] == 0))[0]
                if len(sw_xs) > 0 and abs(sw_xs[-1] - exp_lx) < 60:
                    left_x[y] = float(sw_xs[-1])

        # BIEN PHAI
        if len(r_xs) > 0:
            cur_rx = float(r_xs[-1])
            is_occluded = False
            if len(v_xs) > 0 and (v_xs[-1] >= cur_rx - 35):
                is_occluded = True
            if cur_rx < exp_rx - 45:
                is_occluded = True
            
            # Khong vuot qua via he
            sw_xs = np.where((mask_sw[y] > 0) & (veh_dil[y] == 0))[0]
            sw_right = sw_xs[sw_xs > w * 0.45] if len(sw_xs) else []
            if len(sw_right) > 0 and cur_rx > sw_right[0]:
                cur_rx = float(sw_right[0])
            if cur_rx > exp_rx + 45:
                cur_rx = min(cur_rx, exp_rx + 40)

            if not is_occluded:
                right_x[y] = cur_rx
            else:
                if len(sw_right) > 0 and abs(sw_right[0] - exp_rx) < 60:
                    right_x[y] = float(sw_right[0])

    valid_l = np.where(left_x >= 0)[0]
    valid_l = valid_l[(valid_l >= y_top) & (valid_l <= y_bot)]
    if len(valid_l) >= 2:
        interp_left = np.interp(all_ys, valid_l, left_x[valid_l])
    else:
        interp_left = np.clip(la * all_ys + lb, 0, w - 1)

    valid_r = np.where(right_x >= 0)[0]
    valid_r = valid_r[(valid_r >= y_top) & (valid_r <= y_bot)]
    if len(valid_r) >= 2:
        interp_right = np.interp(all_ys, valid_r, right_x[valid_r])
    else:
        interp_right = np.clip(ra * all_ys + rb, 0, w - 1)

    for idx, y in enumerate(all_ys):
        yl = la * y + lb
        cl = interp_left[idx]
        if yl < cl - 20:
            x1 = max(0, int(yl))
            x2 = min(w, int(cl))
            if x2 > x1 and np.mean(mask_sw[y, x1:x2]) < 0.35:
                interp_left[idx] = yl
        elif cl < yl - 45:
            interp_left[idx] = max(cl, yl - 40)

        yr = ra * y + rb
        cr = interp_right[idx]
        if yr > cr + 20:
            x1 = max(0, int(cr))
            x2 = min(w, int(yr))
            if x2 > x1 and np.mean(mask_sw[y, x1:x2]) < 0.35:
                interp_right[idx] = yr
        elif cr > yr + 45:
            interp_right[idx] = min(cr, yr + 40)

    ksize = 61
    sigma = 18.0
    kg = cv2.getGaussianKernel(ksize, sigma).flatten()
    half = ksize // 2
    pad_l = np.pad(interp_left, half, mode="edge")
    pad_r = np.pad(interp_right, half, mode="edge")
    smooth_l = np.convolve(pad_l, kg, mode="valid")
    smooth_r = np.convolve(pad_r, kg, mode="valid")

    if prev_left_curve is not None and len(prev_left_curve) == len(smooth_l):
        smooth_l = alpha * smooth_l + (1.0 - alpha) * prev_left_curve
    if prev_right_curve is not None and len(prev_right_curve) == len(smooth_r):
        smooth_r = alpha * smooth_r + (1.0 - alpha) * prev_right_curve

    smooth_l = np.minimum(smooth_l, smooth_r - 20.0)
    pts_l = np.column_stack([np.clip(smooth_l, 0, w - 1).astype(np.int32), all_ys])
    pts_r = np.column_stack([np.clip(smooth_r, 0, w - 1).astype(np.int32), all_ys])
    poly = np.vstack([pts_l, pts_r[::-1]])
    legal_corridor_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(legal_corridor_poly, [poly], 255)

    left_x_by_y = np.zeros(h, dtype=np.float32)
    right_x_by_y = np.full(h, w - 1, dtype=np.float32)
    left_x_by_y[:y_top] = smooth_l[0]
    left_x_by_y[y_top:y_bot + 1] = smooth_l
    left_x_by_y[y_bot + 1:] = smooth_l[-1]
    right_x_by_y[:y_top] = smooth_r[0]
    right_x_by_y[y_top:y_bot + 1] = smooth_r
    right_x_by_y[y_bot + 1:] = smooth_r[-1]

    return legal_corridor_poly, pts_l, pts_r, left_x_by_y, right_x_by_y, smooth_l, smooth_r

def test_corner_artifact(cnt, width, height, y_top, y_bottom):
    pts = cnt.reshape(-1, 2)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    y_threshold = y_top + int((y_bottom - y_top) * 0.65)

    if x_min <= 130 and y_max >= (y_bottom - 25) and y_min >= y_threshold:
        return True
    if x_min <= 80 and y_max >= int(height * 0.88):
        return True
    if x_max >= (width - 130) and y_max >= (y_bottom - 25) and y_min >= y_threshold:
        return True
    if x_max >= (width - 80) and y_max >= int(height * 0.88):
        return True
    if x_min >= (width - 240) and y_max >= int(height * 0.82) and (y_max - y_min) < 130:
        return True
    return False

# Now let us test on Incident 5 frames (1780-1810) and Incident 6 frames (2025-2045)
print("=== Testing on frames 1785 to 1810 ===")
prev_yolo_l, prev_yolo_r = None, None
prev_cl, prev_cr = None, None
alerts_1785 = 0

for f_idx in range(1785, 1811):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret, frame_bgr = cap.read()
    h, w = frame_bgr.shape[:2]
    
    # YOLO
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)
        
    inp = cv2.resize(frame_bgr, (2048, 1024))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device).half()
    with torch.no_grad():
        out = pidnet_model(inp)
        if isinstance(out, (list, tuple)): out = out[1]
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        pred = out.argmax(dim=1)[0].cpu().numpy()
        
    mask_sw = (pred == 1) | (yolo_pred == 1)
    mask_road = (pred == 0) & (~mask_sw)
    veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)

    (la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(
        yolo_pred, h, w, road_id=0, sidewalk_id=1,
        prev_left=prev_yolo_l, prev_right=prev_yolo_r, alpha=0.15
    )
    prev_yolo_l, prev_yolo_r = (la, lb), (ra, rb)
    y_top = int(h * 0.35)
    y_bot = int(h * 0.98)
    
    corr_poly, pts_l, pts_r, lx_by_y, rx_by_y, prev_cl, prev_cr = extract_curved_corridor_clamped(
        mask_road, mask_sw, veh_mask, h, w, y_top, y_bot,
        prev_left_curve=prev_cl, prev_right_curve=prev_cr, alpha=0.15,
        yolo_lines=((la, lb), (ra, rb))
    )
    
    normal_classes = [0, 1] + [k for k, v in {0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}.items() if v in CORRIDOR_CLASSES]
    normal_mask = np.isin(pred, normal_classes).astype(np.uint8) * 255
    normal_mask = cv2.morphologyEx(normal_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    road_mask_pure = (pred == 0).astype(np.uint8)
    road_dilated = cv2.dilate(road_mask_pure, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    horizon_cutoff = max(int(h * 0.50), y_top + 65)
    is_road_level = (np.arange(h)[:, None] >= horizon_cutoff)
    is_vehicle = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
    
    anomaly_core = (
        (corr_poly == 255) & (normal_mask == 0) & is_road_level & (road_dilated > 0) & (~is_vehicle)
    )
    obs_closed = cv2.morphologyEx(anomaly_core.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    encroach_mask = cv2.morphologyEx(obs_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cand = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if area < 600: continue
        if test_corner_artifact(cnt, w, h, y_top, y_bot): continue
        cnt_side = "Trái" if (bx + bw * 0.35) < (w * 0.5) else "Phải"
        mid_y = int(np.clip(by + bh * 0.5, 0, h - 1))
        curb_x = rx_by_y[mid_y] if cnt_side == "Phải" else lx_by_y[mid_y]
        if cnt_side == "Trái" and (bx - curb_x) > 45: continue
        if cnt_side == "Phải" and (curb_x - (bx + bw)) > 45: continue
        cand.append((bx, by, bw, bh, area))
        
    if len(cand) > 0:
        print(f"Frame {f_idx}: {len(cand)} candidates -> {cand}")
        alerts_1785 += 1

print(f"Total alert frames in 1785-1810: {alerts_1785} (was 13 before!)")

print("\n=== Testing on frames 2025 to 2050 ===")
prev_cl, prev_cr = None, None
alerts_2025 = 0
for f_idx in range(2025, 2051):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret, frame_bgr = cap.read()
    h, w = frame_bgr.shape[:2]
    
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)
        
    inp = cv2.resize(frame_bgr, (2048, 1024))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device).half()
    with torch.no_grad():
        out = pidnet_model(inp)
        if isinstance(out, (list, tuple)): out = out[1]
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        pred = out.argmax(dim=1)[0].cpu().numpy()
        
    mask_sw = (pred == 1) | (yolo_pred == 1)
    mask_road = (pred == 0) & (~mask_sw)
    veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)

    (la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(
        yolo_pred, h, w, road_id=0, sidewalk_id=1,
        prev_left=prev_yolo_l, prev_right=prev_yolo_r, alpha=0.15
    )
    prev_yolo_l, prev_yolo_r = (la, lb), (ra, rb)
    y_top = int(h * 0.35)
    y_bot = int(h * 0.98)
    
    corr_poly, pts_l, pts_r, lx_by_y, rx_by_y, prev_cl, prev_cr = extract_curved_corridor_clamped(
        mask_road, mask_sw, veh_mask, h, w, y_top, y_bot,
        prev_left_curve=prev_cl, prev_right_curve=prev_cr, alpha=0.15,
        yolo_lines=((la, lb), (ra, rb))
    )
    
    normal_classes = [0, 1] + [k for k, v in {0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car", 14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}.items() if v in CORRIDOR_CLASSES]
    normal_mask = np.isin(pred, normal_classes).astype(np.uint8) * 255
    normal_mask = cv2.morphologyEx(normal_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    road_mask_pure = (pred == 0).astype(np.uint8)
    road_dilated = cv2.dilate(road_mask_pure, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))
    horizon_cutoff = max(int(h * 0.50), y_top + 65)
    is_road_level = (np.arange(h)[:, None] >= horizon_cutoff)
    is_vehicle = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
    
    anomaly_core = (
        (corr_poly == 255) & (normal_mask == 0) & is_road_level & (road_dilated > 0) & (~is_vehicle)
    )
    obs_closed = cv2.morphologyEx(anomaly_core.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    encroach_mask = cv2.morphologyEx(obs_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cand = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if area < 600: continue
        if test_corner_artifact(cnt, w, h, y_top, y_bot): continue
        cnt_side = "Trái" if (bx + bw * 0.35) < (w * 0.5) else "Phải"
        mid_y = int(np.clip(by + bh * 0.5, 0, h - 1))
        curb_x = rx_by_y[mid_y] if cnt_side == "Phải" else lx_by_y[mid_y]
        if cnt_side == "Trái" and (bx - curb_x) > 45: continue
        if cnt_side == "Phải" and (curb_x - (bx + bw)) > 45: continue
        cand.append((bx, by, bw, bh, area))
        
    if len(cand) > 0:
        print(f"Frame {f_idx}: {len(cand)} candidates -> {cand}")
        alerts_2025 += 1

print(f"Total alert frames in 2025-2050: {alerts_2025}")
cap.release()
