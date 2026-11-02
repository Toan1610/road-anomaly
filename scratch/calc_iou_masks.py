import cv2
import numpy as np

cap_ov = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_enc = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")

test_frames = [116, 322, 986, 1786, 1790, 2039]

for f in test_frames:
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret1, ov = cap_ov.read()
    cap_enc.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret2, enc = cap_enc.read()
    
    if not (ret1 and ret2): continue
    
    # In encroach_mask, white is > 128
    enc_mask = (enc[:, :, 0] > 128).astype(np.uint8)
    
    # In overlay, green road has dominant green:
    # COLORS['road'] in BGR: (85, 180, 34).
    # blended = 0.65 * orig + 0.35 * (85, 180, 34)
    # A reliable way to identify road overlay: G channel significantly higher than R & B,
    # or check where G > 100 and G > R + 25 and G > B + 25
    green_road = ((ov[:, :, 1] > 90) & (ov[:, :, 1] > ov[:, :, 0].astype(int) + 20) & (ov[:, :, 1] > ov[:, :, 2].astype(int) + 20)).astype(np.uint8)
    
    intersection = np.sum((enc_mask == 1) & (green_road == 1))
    union = np.sum((enc_mask == 1) | (green_road == 1))
    iou = intersection / max(1, union)
    
    enc_px = np.sum(enc_mask == 1)
    grn_px = np.sum(green_road == 1)
    
    # Area in enc but not in green_road (white corridor extending outside green road, e.g. into sidewalk/wall)
    enc_only = np.sum((enc_mask == 1) & (green_road == 0))
    # Area in green_road but not in enc
    grn_only = np.sum((enc_mask == 0) & (green_road == 1))
    
    print(f"Frame {f:4d}: IoU={iou:.3f} | EncroachMask px={enc_px:7d} | OverlayGreen px={grn_px:7d} | EncroachOnly (Corridor overspill)={enc_only:7d} ({enc_only/max(1, enc_px)*100:.1f}%) | GreenOnly={grn_only:7d}")

cap_ov.release()
cap_enc.release()
