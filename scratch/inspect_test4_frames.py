import cv2
import numpy as np

# Let's inspect video/test4.mp4, output_test4_pidnet/overlay.mp4, and output_test4_pidnet/encroach_mask.mp4
cap_test4 = cv2.VideoCapture("video/test4.mp4")
cap_overlay = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_encroach = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")

# Look at frames around Incident 5 (f=1786) and Incident 6 (f=2039)
# Also look at Incident 4 (f=1790) and Incident 2 (f=322)
target_frames = [322, 986, 1780, 1786, 1790, 2035, 2039, 2045]

for f_idx in target_frames:
    cap_test4.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret_orig, orig = cap_test4.read()
    
    cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret_ov, ov = cap_overlay.read()
    
    cap_encroach.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret_enc, enc = cap_encroach.read()
    
    if ret_orig and ret_ov and ret_enc:
        # Resize to 960x540
        h, w = orig.shape[:2]
        s_orig = cv2.resize(orig, (w // 2, h // 2))
        s_ov = cv2.resize(ov, (w // 2, h // 2))
        s_enc = cv2.resize(enc, (w // 2, h // 2))
        
        # Combine horizontally
        row = np.hstack([s_orig, s_ov, s_enc])
        cv2.imwrite(f"scratch/debug_f{f_idx:04d}.jpg", row)
        print(f"Frame {f_idx}: wrote scratch/debug_f{f_idx:04d}.jpg")

cap_test4.release()
cap_overlay.release()
cap_encroach.release()
