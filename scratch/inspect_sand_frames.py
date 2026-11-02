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
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Let's inspect frames 95..128
frames = {}
cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
f_idx = 90
while cap.isOpened() and f_idx <= 130:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frames[f_idx] = frame_bgr.copy()
    f_idx += 1
cap.release()

print(f"Analyzing frames 95..130 ({len(frames)} frames)...")
prev_probs = None
for fid in range(95, 128):
    frame_bgr = frames[fid]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # YOLO
    yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
    yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

    # SegFormer
    inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = seg_model(**inputs)
        logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        probs = torch.softmax(logits[0], dim=0)
        pred = probs.argmax(dim=0).cpu().numpy()

    # Vehicles
    vehicle_ids = [k for k, v in seg_model.config.id2label.items() if v in {"car", "truck", "bus", "motorcycle", "bicycle", "rider", "person"}]
    vehicle_mask = np.isin(pred, vehicle_ids).astype(np.uint8) * 255
    vehicle_dilated = cv2.dilate(vehicle_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))

    # Check sand pile position (it is around y in [420..600], x in [250..550])
    # Where is building class (sand pile is class 2 in SegFormer)
    bld_mask = (pred == 2).astype(np.uint8)
    # Check overlap with vehicle_dilated
    cnts, _ = cv2.findContours(bld_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sand_cnt = None
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if by > 350 and bx < 600 and cv2.contourArea(c) > 200:
            sand_cnt = c
            break
            
    if sand_cnt is not None:
        bx, by, bw, bh = cv2.boundingRect(sand_cnt)
        c_area = cv2.contourArea(sand_cnt)
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [sand_cnt], -1, 1, -1)
        ov = np.sum((c_mask == 1) & (vehicle_dilated == 255))
        ov_ratio = ov / max(1, np.sum(c_mask == 1))
        print(f"Frame {fid:3d}: Sand cnt bbox=({bx},{by},{bw},{bh}), area={c_area:.0f}, vehicle_overlap={ov_ratio*100:.1f}%")
    else:
        print(f"Frame {fid:3d}: Sand cnt NOT FOUND in class 2")
