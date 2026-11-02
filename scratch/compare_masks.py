import cv2
import numpy as np

# Open videos
cap_overlay = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_encroach = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")
cap_road = cv2.VideoCapture("output_test4_pidnet/road_mask.mp4")

print("Overlay opened:", cap_overlay.isOpened(), "total frames:", int(cap_overlay.get(cv2.CAP_PROP_FRAME_COUNT)))
print("Encroach opened:", cap_encroach.isOpened(), "total frames:", int(cap_encroach.get(cv2.CAP_PROP_FRAME_COUNT)))
print("Road opened:", cap_road.isOpened(), "total frames:", int(cap_road.get(cv2.CAP_PROP_FRAME_COUNT)))

# Let's inspect some frames: frame 100, 322, 1786, 2039
check_frames = [116, 322, 986, 1786, 1790, 2039]

for target_f in check_frames:
    cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
    ret_ov, f_ov = cap_overlay.read()
    
    cap_encroach.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
    ret_enc, f_enc = cap_encroach.read()
    
    cap_road.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
    ret_rd, f_rd = cap_road.read()
    
    if ret_ov and ret_enc:
        # In overlay, green is road: COLORS['road'] = (34, 180, 85) in RGB -> in BGR: (85, 180, 34)
        # In encroach_mask, white (255) is corridor without obstacle, black (0) is outside or obstacle
        # Save a comparison image
        h, w = f_ov.shape[:2]
        vis = np.zeros((h // 2, w, 3), dtype=np.uint8)
        
        # Resize to half for compact saving
        ov_small = cv2.resize(f_ov, (w // 2, h // 2))
        enc_small = cv2.resize(f_enc, (w // 2, h // 2))
        
        comp = np.hstack([ov_small, enc_small])
        cv2.imwrite(f"scratch/comp_f{target_f:04d}.jpg", comp)
        print(f"Saved scratch/comp_f{target_f:04d}.jpg")

cap_overlay.release()
cap_encroach.release()
cap_road.release()
