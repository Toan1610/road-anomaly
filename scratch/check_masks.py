import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024").to(device).eval()
yolo_model = YOLO("yolo26n-sem.pt")

cap = cv2.VideoCapture("video/test2.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

f_idx = 0
frame_bgr = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if f_idx == 85:
        frame_bgr = frame.copy()
        break
    f_idx += 1
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

print("Scanline analysis for Frame 85:")
print(f"{'y':>5} | {'road xs':>20} | {'sw xs':>20} | {'yolo road xs':>20} | {'yolo sw xs':>20}")
for y in range(250, 700, 25):
    r_xs = np.where(mask_road[y])[0]
    sw_xs = np.where(mask_sw[y])[0]
    yr_xs = np.where(yolo_pred[y] == 0)[0]
    ysw_xs = np.where(yolo_pred[y] == 1)[0]
    
    r_str = f"[{r_xs[0]}, {r_xs[-1]}]" if len(r_xs) else "None"
    sw_str = f"[{sw_xs[0]}, {sw_xs[-1]}]" if len(sw_xs) else "None"
    yr_str = f"[{yr_xs[0]}, {yr_xs[-1]}]" if len(yr_xs) else "None"
    ysw_str = f"[{ysw_xs[0]}, {ysw_xs[-1]}]" if len(ysw_xs) else "None"
    print(f"{y:>5} | {r_str:>20} | {sw_str:>20} | {yr_str:>20} | {ysw_str:>20}")
