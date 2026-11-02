import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
hf_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
proc = SegformerImageProcessor.from_pretrained(hf_name)
model = SegformerForSemanticSegmentation.from_pretrained(hf_name).to(device).eval()

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 105)
ret, frame_bgr = cap.read()
h, w = frame_bgr.shape[:2]

inputs = proc(images=Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs)
pred = torch.nn.functional.interpolate(out.logits, size=(h, w), mode="bilinear", align_corners=False).argmax(dim=1)[0].cpu().numpy()

# Classes in Cityscapes:
# 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole, 6: traffic light, 7: traffic sign
# 8: vegetation, 9: terrain, 10: sky, 11: person, 12: rider, 13: car, 14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle

mask_road = (pred == 0).astype(np.uint8) * 255
mask_sw = (pred == 1).astype(np.uint8) * 255
mask_vehicles = np.isin(pred, [11, 12, 13, 14, 15, 16, 17, 18]).astype(np.uint8) * 255

# Let's inspect the road corridor and curb line
# Sand pile on the left:
# Notice that at y in [450, 600], the left edge of the road is pushed inwards (to the right) by the sand pile!
# Let us inspect the left edge of the road as a function of y
edges_left = []
for y in range(int(h * 0.50), int(h * 0.95), 5):
    road_xs = np.where(mask_road[y, :w // 2] > 0)[0]
    if len(road_xs) > 0:
        edges_left.append((y, road_xs[0]))

print("Left road edge samples (y, x_min):")
for y, x in edges_left[::3]:
    print(f"y={y}: x_left={x}")

cv2.imwrite("scratch/frame105.jpg", frame_bgr)
