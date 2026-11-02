import cv2
import numpy as np

# Check overlay video around frames 1770 to 1805
cap_ov = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")

# We can search for any red pixels in overlay or read HUD text
# Let's inspect frames 1770 to 1805
for f in range(1770, 1810):
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap_ov.read()
    if not ret: break
    
    # Check if HUD says CANH BAO (red)
    # HUD box is at (15, 15) to (660, 105)
    hud_crop = frame[15:60, 15:400]
    # Check red text in HUD: (B, G, R) where R > 180 and G < 50 and B < 50
    red_hud = np.sum((hud_crop[:, :, 2] > 180) & (hud_crop[:, :, 1] < 50) & (hud_crop[:, :, 0] < 50))
    
    # Also check if there is red polygon in frame: (B, G, R) with R > 200, B < 50, G < 50
    red_poly = np.sum((frame[105:, :, 2] > 200) & (frame[105:, :, 1] < 50) & (frame[105:, :, 0] < 50))
    
    if red_hud > 50 or red_poly > 50:
        print(f"Frame {f}: ALERT! red_hud_px={red_hud}, red_poly_px={red_poly}")

print("\n--- Now checking frames 2020 to 2060 ---")
for f in range(2020, 2060):
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = cap_ov.read()
    if not ret: break
    
    hud_crop = frame[15:60, 15:400]
    red_hud = np.sum((hud_crop[:, :, 2] > 180) & (hud_crop[:, :, 1] < 50) & (hud_crop[:, :, 0] < 50))
    red_poly = np.sum((frame[105:, :, 2] > 200) & (frame[105:, :, 1] < 50) & (frame[105:, :, 0] < 50))
    
    if red_hud > 50 or red_poly > 50:
        print(f"Frame {f}: ALERT! red_hud_px={red_hud}, red_poly_px={red_poly}")

cap_ov.release()
