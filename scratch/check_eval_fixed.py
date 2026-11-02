import cv2
import numpy as np

cap_ov = cv2.VideoCapture("output_test4_fix_eval/overlay.mp4")
cap_enc = cv2.VideoCapture("output_test4_fix_eval/encroach_mask.mp4")

total_frames = int(cap_ov.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in eval:", total_frames)

for local_idx in [36, 150, 289]:  # frame 1786 (1750+36), frame 1900, frame 2039 (1750+289)
    cap_ov.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
    ret1, ov = cap_ov.read()
    cap_enc.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
    ret2, enc = cap_enc.read()
    
    if ret1 and ret2:
        enc_mask = (enc[:, :, 0] > 128).astype(np.uint8)
        
        # Save side-by-side comparison
        h, w = ov.shape[:2]
        s_ov = cv2.resize(ov, (w // 2, h // 2))
        s_enc = cv2.resize(enc, (w // 2, h // 2))
        comp = np.hstack([s_ov, s_enc])
        
        real_frame = 1750 + local_idx
        out_p = f"scratch/eval_fixed_f{real_frame:04d}.jpg"
        cv2.imwrite(out_p, comp)
        print(f"Saved {out_p}")

cap_ov.release()
cap_enc.release()
