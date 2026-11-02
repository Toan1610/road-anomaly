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

for target_fid in [98, 140, 199]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_fid)
    ret, frame_bgr = cap.read()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # SegFormer
    inputs = processor(images=Image.fromarray(frame_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = seg_model(**inputs)
        logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = logits[0].argmax(dim=0).cpu().numpy()

    # YOLO
    yolo_res = yolo_model(frame_bgr, imgsz=(736, 1280), device=0, verbose=False)[0]
    yolo_pred = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_sw = (pred == 1) | (yolo_pred == 1)
    mask_road = (pred == 0) & (~mask_sw)
    veh_mask = np.isin(pred, [11, 12, 13, 14, 15, 17, 18])

    (la, lb), (ra, rb) = rve.fit_dynamic_road_lines_yolo(yolo_pred, h, w)
    corridor, pts_l, pts_r, lx_by_y, rx_by_y, _, _ = rve.extract_curved_road_corridor(
        mask_road, mask_sw, veh_mask, h, w, yolo_lines=((la, lb), (ra, rb))
    )

    normal_mask = np.isin(pred, [0, 1] + [k for k, v in seg_model.config.id2label.items() if v in rve.CORRIDOR_CLASSES]).astype(np.uint8) * 255
    road_dilated = cv2.dilate((pred == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    encroach_raw = (corridor == 255) & (normal_mask == 0) & (road_dilated > 0)
    encroach_mask = cv2.morphologyEx(encroach_raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(encroach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_ids = [k for k, v in seg_model.config.id2label.items() if v in {"car", "truck", "bus", "motorcycle", "bicycle", "rider", "person"}]
    vehicle_mask = np.isin(pred, vehicle_ids).astype(np.uint8) * 255

    # Base overlay
    overlay = frame_rgb.copy()
    seg_color = np.zeros_like(overlay)
    seg_color[mask_road] = rve.COLORS["road"]
    seg_color[mask_sw] = rve.COLORS["sidewalk"]
    blended = cv2.addWeighted(overlay, 0.55, seg_color, 0.45, 0)
    overlay[mask_road | mask_sw] = blended[mask_road | mask_sw]

    # Draw refined obstacle contours
    fill_layer = np.zeros_like(overlay)
    for cnt in contours:
        if cv2.contourArea(cnt) > 250:
            obs_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(obs_mask, [cnt], -1, 255, -1)
            obs_mask[vehicle_mask > 0] = 0
            obs_cnts, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if obs_cnts:
                cv2.drawContours(overlay, obs_cnts, -1, (255, 0, 0), 3, cv2.LINE_AA)
            fill_layer[obs_mask == 255] = (255, 50, 0)

    cv2.addWeighted(overlay, 1.0, fill_layer, 0.5, 0, dst=overlay)
    cv2.imwrite(f"scratch/vis_clean_f{target_fid}.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved scratch/vis_clean_f{target_fid}.jpg")

cap.release()
