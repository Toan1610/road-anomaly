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

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 115)
ret, frame_bgr = cap.read()
cap.release()

inputs = processor(images=Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = seg_model(**inputs)
    logits = F.interpolate(outputs.logits, size=(720, 1280), mode="bilinear", align_corners=False)
    pred = logits[0].argmax(dim=0).cpu().numpy()

labels = seg_model.config.id2label
normal_classes = [0, 1] + [k for k, v in labels.items() if v in rve.CORRIDOR_CLASSES]
normal_raw = np.isin(pred, normal_classes).astype(np.uint8) * 255
normal_closed = cv2.morphologyEx(normal_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

road_dilated = cv2.dilate((pred == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

raw_diff = (normal_raw == 0) & (road_dilated > 0)
closed_diff = (normal_closed == 0) & (road_dilated > 0)

print(f"Raw diff pixels: {np.sum(raw_diff)}, Closed diff pixels: {np.sum(closed_diff)}")

# Where are building pixels (sand pile) that touch road_dilated?
bld_touch_road = (pred == 2) & (road_dilated > 0)
print(f"Building touching road: {np.sum(bld_touch_road)} pixels")
if np.sum(bld_touch_road) > 0:
    ys, xs = np.where(bld_touch_road)
    print(f"  y range: {ys.min()}..{ys.max()}, x range: {xs.min()}..{xs.max()}")
