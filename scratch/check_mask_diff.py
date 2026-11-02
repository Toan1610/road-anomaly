import cv2
import numpy as np

# Let's inspect the correlation between overlay road area and encroach_mask white area
cap_ov = cv2.VideoCapture("output_test4_pidnet/overlay.mp4")
cap_enc = cv2.VideoCapture("output_test4_pidnet/encroach_mask.mp4")
cap_rd = cv2.VideoCapture("output_test4_pidnet/road_mask.mp4")

total_frames = int(cap_ov.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total_frames)

# Check 5 representative frames
test_f = [100, 322, 1000, 1786, 2039]

for f in test_f:
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret1, ov = cap_ov.read()
    cap_enc.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret2, enc = cap_enc.read()
    cap_rd.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret3, rd = cap_rd.read()
    
    # In overlay, green blend is:
    # blended = 0.65 * frame + 0.35 * (85, 180, 34) (in BGR)
    # The green channel is boosted.
    # In encroach_mask: white (255, 255, 255) vs black (0, 0, 0)
    enc_gray = cv2.cvtColor(enc, cv2.COLOR_BGR2GRAY)
    rd_gray = cv2.cvtColor(rd, cv2.COLOR_BGR2GRAY)
    
    enc_white_px = np.sum(enc_gray > 128)
    rd_white_px = np.sum(rd_gray > 128)
    enc_black_px_in_rd = np.sum((rd_gray > 128) & (enc_gray <= 128))
    
    print(f"\n--- Frame {f} ---")
    print(f"road_mask white pixels (legal_corridor_poly): {rd_white_px}")
    print(f"encroach_mask white pixels (corridor - obstacle): {enc_white_px}")
    print(f"Obstacle pixels cut out in encroach_mask: {enc_black_px_in_rd}")
    
    # Let's check the shape of legal_corridor_poly:
    # Find bounding box or boundary of rd_gray
    ys, xs = np.where(rd_gray > 128)
    if len(ys) > 0:
        print(f"road_mask Y range: [{ys.min()}, {ys.max()}], X range: [{xs.min()}, {xs.max()}]")
    else:
        print("road_mask is completely empty (black)!")

cap_ov.release()
cap_enc.release()
cap_rd.release()
