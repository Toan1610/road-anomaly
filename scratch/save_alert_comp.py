import cv2
import numpy as np

cap_ov = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_enc = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")

for target_f in [1800, 1805, 2040]:
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
    ret1, ov = cap_ov.read()
    cap_enc.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
    ret2, enc = cap_enc.read()
    
    if ret1 and ret2:
        h, w = ov.shape[:2]
        s_ov = cv2.resize(ov, (w // 2, h // 2))
        s_enc = cv2.resize(enc, (w // 2, h // 2))
        comp = np.hstack([s_ov, s_enc])
        cv2.imwrite(f"scratch/alert_comp_f{target_f:04d}.jpg", comp)
        print("Saved alert_comp for frame", target_f)

cap_ov.release()
cap_enc.release()
