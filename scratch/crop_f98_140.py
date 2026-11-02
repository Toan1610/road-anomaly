import cv2
import numpy as np

cap = cv2.VideoCapture("video/test2.mp4")
for fid in [98, 140]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame_bgr = cap.read()
    if fid == 98:
        # crop around x: 500..650, y: 400..500
        crop = frame_bgr[420:490, 530:640]
        cv2.imwrite("scratch/crop_f98.jpg", crop)
    elif fid == 140:
        crop = frame_bgr[470:540, 450:580]
        cv2.imwrite("scratch/crop_f140.jpg", crop)
cap.release()
print("Saved crops.")
