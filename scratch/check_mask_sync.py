import cv2
import numpy as np

cap_ov = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_enc = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")

# Check several frames across the video: frame 18, frame 100, frame 322, frame 1000, frame 1800
test_frames = [18, 100, 322, 1000, 1800]

for f in test_frames:
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret1, ov = cap_ov.read()
    cap_enc.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret2, enc = cap_enc.read()
    
    if not (ret1 and ret2):
        continue
        
    h, w = ov.shape[:2]
    # Encroach mask: white pixels (value > 128)
    enc_white = (enc[:, :, 0] > 128).astype(np.uint8)
    
    # Overlay: green road pixels:
    # COLORS['road'] in BGR: (85, 180, 34).
    # blended = 0.65 * orig + 0.35 * (85, 180, 34)
    # G channel dominant
    green_road = ((ov[:, :, 1] > 80) & 
                  (ov[:, :, 1] > ov[:, :, 0].astype(int) + 18) & 
                  (ov[:, :, 1] > ov[:, :, 2].astype(int) + 18)).astype(np.uint8)
                  
    intersection = np.sum((enc_white == 1) & (green_road == 1))
    union = np.sum((enc_white == 1) | (green_road == 1))
    iou = intersection / max(1, union)
    
    enc_px = np.sum(enc_white == 1)
    grn_px = np.sum(green_road == 1)
    
    # Difference: enc white where green is 0 (e.g. vehicles on the road)
    enc_no_grn = np.sum((enc_white == 1) & (green_road == 0))
    # Green where enc is 0
    grn_no_enc = np.sum((green_road == 1) & (enc_white == 0))
    
    print(f"Frame {f:4d}: IoU(enc, green) = {iou:.3f} | EncroachWhite={enc_px} | GreenRoad={grn_px} | EncWhite_not_Green={enc_no_grn} | Green_not_EncWhite={grn_no_enc}")
    
    # Save visual comparison: Overlay, EncroachMask, and Diff
    diff_vis = np.zeros_like(ov)
    # Yellow = both agree (road)
    diff_vis[(enc_white == 1) & (green_road == 1)] = (0, 255, 255)
    # White = Encroach only (vehicle on road, etc.)
    diff_vis[(enc_white == 1) & (green_road == 0)] = (255, 255, 255)
    # Green = Green only
    diff_vis[(green_road == 1) & (enc_white == 0)] = (0, 255, 0)
    
    s_ov = cv2.resize(ov, (w // 3, h // 3))
    s_enc = cv2.resize(enc, (w // 3, h // 3))
    s_diff = cv2.resize(diff_vis, (w // 3, h // 3))
    
    row = np.hstack([s_ov, s_enc, s_diff])
    cv2.imwrite(f"scratch/check_sync_f{f:04d}.jpg", row)

cap_ov.release()
cap_enc.release()
print("Saved comparison images to scratch/check_sync_f*.jpg")
