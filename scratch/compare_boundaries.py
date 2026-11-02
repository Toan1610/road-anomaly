import cv2
import torch
import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from run_video_encroachment import (
    load_pidnet_model,
    infer_pidnet_probs,
    extract_curved_road_corridor,
    make_class_lut,
)
from run_video_encroachment_mask2former import infer_mask2former_probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pidnet = load_pidnet_model("pidnet-s", "PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt", device, use_fp16=True)

m2f_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(m2f_name, local_files_only=True)
m2f = Mask2FormerForUniversalSegmentation.from_pretrained(m2f_name, local_files_only=True).to(device)
m2f.eval()

cap = cv2.VideoCapture("video/test2.mp4")
frame_idx = 0
prev_p_l, prev_p_r = None, None
prev_m_l, prev_m_r = None, None

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_idx += 1
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # PIDNet
    p_prob = infer_pidnet_probs(pidnet, frame_rgb, H, W, device, use_fp16=True).squeeze(0).cpu().numpy()
    p_pred = np.argmax(p_prob, axis=0)
    p_road = (p_pred == 0).astype(np.uint8) * 255
    p_sw = (p_pred == 1).astype(np.uint8) * 255
    p_veh = np.isin(p_pred, [11, 12, 13, 14, 15, 16, 17, 18])
    p_infra = np.isin(p_pred, [1, 3, 4, 8, 9]).astype(np.uint8)
    
    _, _, _, p_lx, p_rx, prev_p_l, prev_p_r = extract_curved_road_corridor(
        p_road, p_sw, p_veh, H, W,
        prev_left_curve=prev_p_l, prev_right_curve=prev_p_r,
        mask_infra=p_infra
    )
    
    # Mask2Former
    m_prob = infer_mask2former_probs(m2f, processor, frame_rgb, H, W, device, use_fp16=True).cpu().numpy()
    m_pred = np.argmax(m_prob, axis=0)
    m_road = (m_pred == 0).astype(np.uint8) * 255
    m_sw = (m_pred == 1).astype(np.uint8) * 255
    m_veh = np.isin(m_pred, [11, 12, 13, 14, 15, 16, 17])
    m_infra = np.isin(m_pred, [1, 3, 4, 8, 9]).astype(np.uint8)
    
    _, _, _, m_lx, m_rx, prev_m_l, prev_m_r = extract_curved_road_corridor(
        m_road, m_sw, m_veh, H, W,
        prev_left_curve=prev_m_l, prev_right_curve=prev_m_r,
        mask_infra=m_infra
    )
    
    if frame_idx in (50, 100, 150, 190):
        print(f"=== Frame {frame_idx} ===")
        print(f"  PIDNet:       left(550)={p_lx[550]:.1f}, right(550)={p_rx[550]:.1f}, road_r_max={np.where(p_road[550]>0)[0][-1] if np.any(p_road[550]>0) else -1}")
        print(f"  Mask2Former:  left(550)={m_lx[550]:.1f}, right(550)={m_rx[550]:.1f}, road_r_max={np.where(m_road[550]>0)[0][-1] if np.any(m_road[550]>0) else -1}")

cap.release()
