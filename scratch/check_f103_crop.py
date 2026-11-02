import cv2
import numpy as np

# Load Frame 103 and check what obs_mask and obs_cnts look like
cap = cv2.VideoCapture("output_test2/overlay.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 103)
ret, frame = cap.read()
cap.release()

# Crop around the red contour
crop = frame[400:500, 400:600]
cv2.imwrite("scratch/crop_f103_red.jpg", crop)
print("Saved scratch/crop_f103_red.jpg")
