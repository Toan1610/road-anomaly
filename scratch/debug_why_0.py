import cv2
import torch
import numpy as np

# Let's import run_video_encroachment_mask2former and run exactly its loop logic on frame 190
from run_video_encroachment_mask2former import *

class Args:
    pass

args = parse_args()
args.input = "video/test2.mp4"
args.output = "scratch/test_out"
args.start_frame = 188
args.max_frames = 194

# Run main with modified print
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 188)
frame_idx = 188
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
road_id = 0
sidewalk_id = 1
normal_classes = [road_id, sidewalk_id] + [k for k, v in labels.items() if v in CORRIDOR_CLASSES]
LUT_NORMAL = make_class_lut(normal_classes)
gpu_morph = GpuMorphology(device)
tracker = EncroachTracker(fps=fps, min_duration_sec=0.8)

prev_curve_left = None
prev_curve_right = None
y_top = int(H * 0.35)
y_bottom = int(H * 0.95)

for _ in range(5):
    ret, frame_bgr = cap.read()
    if not ret: break
    frame_idx += 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Exactly as in run_video_encroachment_mask2former.py:
    probs = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, True)
    pred = probs.argmax(dim=0).cpu().numpy()
    pred_u8 = pred.clip(0, 255).astype(np.uint8)
    
    mask_sw = (pred == sidewalk_id)
    mask_infra = LUT_INFRA[pred_u8].astype(np.uint8)
    mask_road = (pred == road_id) & (~mask_sw)
    veh_mask = LUT_VEHICLE[pred_u8] & (~mask_sw)
    
    (
        legal_corridor_poly,
        pts_left_curve,
        pts_right_curve,
        left_x_by_y,
        right_x_by_y,
        prev_curve_left,
        prev_curve_right,
    ) = extract_curved_road_corridor(
        mask_road,
        mask_sw,
        veh_mask,
        H,
        W,
        y_top=y_top,
        y_bot=y_bottom,
        prev_left_curve=prev_curve_left,
        prev_right_curve=prev_curve_right,
        alpha=0.15,
        mask_infra=mask_infra,
    )
    
    normal_mask = (LUT_NORMAL[pred_u8]).astype(np.uint8) * 255
    normal_mask = gpu_morph.close(normal_mask, iterations=7)
    
    road_mask_pure = (pred == road_id).astype(np.uint8)
    road_dilated = gpu_morph.dilate(road_mask_pure, iterations=30)
    
    horizon_cutoff = max(int(H * 0.50), y_top + 65)
    is_road_level = (np.arange(H)[:, None] >= horizon_cutoff)
    
    anomaly_core = (
        (legal_corridor_poly == 255)
        & (normal_mask == 0)
        & is_road_level
        & (road_dilated > 0)
        & (~veh_mask)
    )
    
    obs_closed = gpu_morph.close(anomaly_core.astype(np.uint8) * 255, iterations=4)
    encroach_mask = gpu_morph.open(obs_closed, iterations=3)
    
    contours, _ = cv2.findContours(
        encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    print(f"Frame {frame_idx}: contours count={len(contours)}")
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        cnt_side = "Trái" if (bx + bw * 0.35) < (W * 0.5) else "Phải"
        mid_y = int(np.clip(by + bh * 0.5, 0, H - 1))
        curb_x = left_x_by_y[mid_y] if cnt_side == "Trái" else right_x_by_y[mid_y]
        diff = (bx - curb_x) if cnt_side == "Trái" else (curb_x - (bx + bw))
        print(f"  Contour {i}: area={area}, bbox=({bx},{by},{bw},{bh}), side={cnt_side}, curb_x={curb_x:.1f}, diff={diff:.1f}")

