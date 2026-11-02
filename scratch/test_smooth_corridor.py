import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO
import sys
sys.path.insert(0, ".")
import run_video_encroachment as rve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()
yolo_model = YOLO("yolo26n-sem.pt")

cap = cv2.VideoCapture("video/test2.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 85)
ret, frame_bgr = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = seg_model(**inputs)
    logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy()

yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
if yolo_pred.shape != (h, w):
    yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

mask_sw = (pred == 1) | (yolo_pred == 1)
mask_road = (pred == 0) & (~mask_sw)

# Vehicles
veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18]) & (~mask_sw)
veh_dil = cv2.dilate(veh_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

cnts, _ = cv2.findContours(mask_road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clean_road = np.zeros((h, w), dtype=np.uint8)
if cnts:
    max_area = max(cv2.contourArea(c) for c in cnts)
    for c in cnts:
        if cv2.contourArea(c) > max(1000.0, max_area * 0.05):
            cv2.drawContours(clean_road, [c], -1, 255, -1)

y_top = int(h * 0.35)
y_bot = int(h * 0.98)
all_ys = np.arange(y_top, y_bot + 1)

(la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)

# Function to reconstruct boundaries with occlusion rejection
def build_smooth_boundaries(clean_road, veh_dil, mask_sw, y_top, y_bot, la, lb, ra, rb):
    all_ys = np.arange(y_top, y_bot + 1)
    left_x = np.full(h, -1.0, dtype=np.float32)
    right_x = np.full(h, -1.0, dtype=np.float32)

    # 1. Anchor top points using YOLO perspective lines to prevent top flaring
    # The road vanishes in perspective, so at y_top the road width is naturally bounded
    top_anchor_ys = np.arange(y_top, min(y_top + 25, y_bot))
    for ty in top_anchor_ys:
        left_x[ty] = np.clip(la * ty + lb, 0, w * 0.5)
        right_x[ty] = np.clip(ra * ty + rb, w * 0.5, w - 1)

    for y in all_ys:
        if y in top_anchor_ys:
            continue
        r_xs = np.where(clean_road[y] > 0)[0]
        v_xs = np.where(veh_dil[y] > 0)[0]
        exp_lx = la * y + lb
        exp_rx = ra * y + rb

        # --- LEFT SIDE ---
        if len(r_xs) > 0:
            cur_lx = float(r_xs[0])
            is_occluded = False
            # If vehicle pixels exist at or to the left of cur_lx
            if len(v_xs) > 0 and (v_xs[0] <= cur_lx + 35):
                # Vehicle touches or is to the left of road edge
                is_occluded = True
            # If cur_lx is way to the right of expected curb line (> 50px inward)
            if cur_lx > exp_lx + 45:
                is_occluded = True
            
            if not is_occluded:
                left_x[y] = cur_lx
            else:
                # Can we find visible sidewalk curb to the left of vehicles?
                sw_xs = np.where((mask_sw[y] > 0) & (veh_dil[y] == 0))[0]
                if len(sw_xs) > 0 and abs(sw_xs[-1] - exp_lx) < 60:
                    left_x[y] = float(sw_xs[-1])

        # --- RIGHT SIDE ---
        if len(r_xs) > 0:
            cur_rx = float(r_xs[-1])
            is_occluded = False
            if len(v_xs) > 0 and (v_xs[-1] >= cur_rx - 35):
                is_occluded = True
            if cur_rx < exp_rx - 45:
                is_occluded = True
                
            if not is_occluded:
                right_x[y] = cur_rx
            else:
                sw_xs = np.where((mask_sw[y] > 0) & (veh_dil[y] == 0))[0]
                sw_right = sw_xs[sw_xs > w * 0.5] if len(sw_xs) else []
                if len(sw_right) > 0 and abs(sw_right[0] - exp_rx) < 60:
                    right_x[y] = float(sw_right[0])

    # Interpolate across occluded segments
    valid_l = np.where(left_x >= 0)[0]
    valid_l = valid_l[(valid_l >= y_top) & (valid_l <= y_bot)]
    if len(valid_l) >= 2:
        interp_l = np.interp(all_ys, valid_l, left_x[valid_l])
    else:
        interp_l = np.clip(la * all_ys + lb, 0, w - 1)

    valid_r = np.where(right_x >= 0)[0]
    valid_r = valid_r[(valid_r >= y_top) & (valid_r <= y_bot)]
    if len(valid_r) >= 2:
        interp_r = np.interp(all_ys, valid_r, right_x[valid_r])
    else:
        interp_r = np.clip(ra * all_ys + rb, 0, w - 1)

    # Smooth with Gaussian
    ksize = 61
    sigma = 18.0
    kg = cv2.getGaussianKernel(ksize, sigma).flatten()
    half = ksize // 2
    pad_l = np.pad(interp_l, half, mode="edge")
    pad_r = np.pad(interp_r, half, mode="edge")
    smooth_l = np.convolve(pad_l, kg, mode="valid")
    smooth_r = np.convolve(pad_r, kg, mode="valid")
    smooth_l = np.minimum(smooth_l, smooth_r - 20.0)

    # Build corridor mask
    pts_l = np.column_stack([np.clip(smooth_l, 0, w - 1).astype(np.int32), all_ys])
    pts_r = np.column_stack([np.clip(smooth_r, 0, w - 1).astype(np.int32), all_ys])
    poly = np.vstack([pts_l, pts_r[::-1]])
    corridor_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(corridor_mask, [poly], 255)
    return corridor_mask, smooth_l, smooth_r, pts_l, pts_r

corridor_new, sm_l, sm_r, pts_l, pts_r = build_smooth_boundaries(clean_road, veh_dil, mask_sw, y_top, y_bot, la, lb, ra, rb)
cv2.imwrite("scratch/test_corridor_f85.png", corridor_new)

# Overlay on image with red boundary line like user's drawing!
vis = frame_bgr.copy()
# Draw road mask in green
vis[corridor_new == 255] = cv2.addWeighted(vis[corridor_new == 255], 0.6, np.full_like(vis[corridor_new == 255], (0, 220, 0)), 0.4, 0)
# Draw the reconstructed boundary lines in RED
cv2.polylines(vis, [pts_l], False, (0, 0, 255), 4, cv2.LINE_AA)
cv2.polylines(vis, [pts_r], False, (0, 0, 255), 4, cv2.LINE_AA)
cv2.imwrite("scratch/test_vis_f85.jpg", vis)
print("Saved scratch/test_corridor_f85.png and scratch/test_vis_f85.jpg")

print(f"Comparison of Left boundary x at scanlines (Orig vs New):")
orig_corridor = cv2.imread("scratch/corridor_orig_f85.png", cv2.IMREAD_GRAYSCALE)
for y in range(y_top, y_bot, 30):
    idx = y - y_top
    orig_xs = np.where(orig_corridor[y] > 0)[0]
    orig_l = orig_xs[0] if len(orig_xs) else -1
    print(f"y={y:>3d}: Orig left_x={orig_l:>5.1f} | New left_x={sm_l[idx]:>5.1f}")
