import cv2
import numpy as np
import torch
from ultralytics import YOLO
from run_video_encroachment import (
    load_pidnet_model, infer_pidnet_probs, MODEL_CHOICES,
    fit_dynamic_road_lines_yolo, compute_corridor_polygon, is_corner_artifact
)

def test_pipeline():
    device = torch.device("cuda")
    seg_model = load_pidnet_model("pidnet-s", MODEL_CHOICES["pidnet-s"], device, use_fp16=True)
    yolo_model = YOLO("yolo26n-sem.pt")

    cap = cv2.VideoCapture("video/test2.mp4")
    
    # Run across frames 130 to 145
    results = []
    prev_smooth_mask = None

    for f in range(130, 145):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame_bgr = cap.read()
        if not ret: break
        h, w = frame_bgr.shape[:2]

        yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
        yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
        if yolo_pred.shape != (h, w):
            yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)
        probs = infer_pidnet_probs(seg_model, frame_bgr, h, w, device, use_fp16=True)
        pred = probs.argmax(dim=0).cpu().numpy()

        (la, lb), (ra, rb) = fit_dynamic_road_lines_yolo(yolo_pred, h, w)
        y_top = int(h * 0.40)
        y_bottom = int(h * 0.94)
        legal_corridor_poly, _, _ = compute_corridor_polygon(la, lb, ra, rb, y_top, y_bottom, w, h)

        veh_ids = [11, 12, 13, 14, 15, 17, 18]
        veh_mask = np.isin(pred, veh_ids).astype(np.uint8) * 255
        yolo_veh = np.isin(yolo_pred, veh_ids).astype(np.uint8) * 255
        veh_mask = cv2.bitwise_or(veh_mask, yolo_veh)
        veh_dilated = cv2.dilate(veh_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))

        # Cutoff horizon near top of corridor to prevent distant background leakage
        horizon_cutoff = y_top + 35
        valid_y_mask = np.arange(h)[:, None] >= horizon_cutoff

        # Center line to avoid cross-road bridging
        center_x = (la * np.arange(h) + lb + ra * np.arange(h) + rb) * 0.5
        is_left_side = np.arange(w)[None, :] <= (center_x[:, None] + 50)

        # Anomaly core
        normal_mask = np.isin(pred, [0, 1] + veh_ids).astype(np.uint8) * 255
        road_mask_pure = (pred == 0).astype(np.uint8)
        road_dilated = cv2.dilate(road_mask_pure, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35)))

        anomaly_core = (legal_corridor_poly == 255) & (normal_mask == 0) & valid_y_mask & is_left_side
        if veh_dilated is not None:
            anomaly_core = anomaly_core & (veh_dilated == 0)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sand_color = (gray > 125) & (legal_corridor_poly == 255) & valid_y_mask & is_left_side & (veh_dilated == 0)

        obs_expanded = anomaly_core.astype(np.uint8)
        for _ in range(3):
            obs_expanded = (cv2.dilate(obs_expanded, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))) & sand_color.astype(np.uint8)) | obs_expanded

        # Close holes and open noise
        obs_closed = cv2.morphologyEx(obs_expanded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        encroach_mask = cv2.morphologyEx(obs_closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        cnts, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a > 400 and not is_corner_artifact(c, w, h, y_top, y_bottom):
                valid_cnts.append(c)

        # Single solid mask
        cur_obs_mask = np.zeros((h, w), dtype=np.uint8)
        if valid_cnts:
            # Sort by area and keep largest
            main_cnt = max(valid_cnts, key=cv2.contourArea)
            cv2.drawContours(cur_obs_mask, [main_cnt], -1, 255, -1)

        # Temporal EMA smoothing on mask
        if prev_smooth_mask is None:
            smooth_mask_f = cur_obs_mask.astype(np.float32)
        else:
            smooth_mask_f = 0.65 * cur_obs_mask.astype(np.float32) + 0.35 * prev_smooth_mask
        prev_smooth_mask = smooth_mask_f.copy()

        binary_smooth = (smooth_mask_f > 100).astype(np.uint8)
        cnts_smooth, _ = cv2.findContours(binary_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts_smooth:
            main_c = cnts_smooth[0]
            bx, by, bw, bh = cv2.boundingRect(main_c)
            area = cv2.contourArea(main_c)
            print(f"Frame {f}: Solid smooth obstacle area={area:.0f} | bbox=(x={bx}, y={by}, w={bw}, h={bh})")
        else:
            print(f"Frame {f}: No obstacle")

    cap.release()

if __name__ == "__main__":
    test_pipeline()
