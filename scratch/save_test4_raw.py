import cv2
import numpy as np

cap = cv2.VideoCapture("video/test4.mp4")

for f_idx in [100, 322, 1786, 2039]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx - 1)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"scratch/test4_raw_f{f_idx:04d}.jpg", frame)
        # also save crop of right side
        crop_right = frame[600:1080, 1300:1920]
        cv2.imwrite(f"scratch/test4_crop_r_f{f_idx:04d}.jpg", crop_right)

cap.release()
print("Saved raw frames and crops")
