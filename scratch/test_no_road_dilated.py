import cv2
import torch
import numpy as np

for f in [184, 190, 192]:
    road = cv2.imread(f'scratch/masks_debug/f{f}_road.png', 0)
    corridor = cv2.imread(f'scratch/masks_debug/f{f}_corridor.png', 0)
    normal = cv2.imread(f'scratch/masks_debug/f{f}_normal.png', 0)
    
    H, W = road.shape[:2]
    y_top = int(H * 0.35)
    horizon_cutoff = max(int(H * 0.50), y_top + 65)
    is_road_level = (np.arange(H)[:, None] >= horizon_cutoff)
    
    # Without (road_dilated > 0)
    anomaly_core = (
        (corridor == 255)
        & (normal == 0)
        & is_road_level
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    obs_closed = cv2.morphologyEx(anomaly_core.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=3)
    encroach_mask = cv2.morphologyEx(obs_closed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"=== Frame {f} without road_dilated ===")
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < 600:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        print(f"  Contour {i}: area={area}, bbox=({bx},{by},{bw},{bh})")
