import cv2
import numpy as np

# Inspect frame 18 and frame 322
cap_orig = cv2.VideoCapture("video/test4.mp4")
cap_ov = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_enc = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")

for f in [18, 25, 30, 315, 322, 330]:
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret0, orig = cap_orig.read()
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret1, ov = cap_ov.read()
    cap_enc.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret2, enc = cap_enc.read()
    
    if ret0 and ret1 and ret2:
        h, w = orig.shape[:2]
        s_orig = cv2.resize(orig, (w // 2, h // 2))
        s_ov = cv2.resize(ov, (w // 2, h // 2))
        s_enc = cv2.resize(enc, (w // 2, h // 2))
        
        row = np.hstack([s_orig, s_ov, s_enc])
        cv2.imwrite(f"scratch/f{f:04d}_detail.jpg", row)
        print(f"Saved scratch/f{f:04d}_detail.jpg")

cap_orig.release()
cap_ov.release()
cap_enc.release()
