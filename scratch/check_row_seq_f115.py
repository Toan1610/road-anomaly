import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

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

# Where is the sand pile on frame 115?
# Let's save a crop or inspect rows 450 to 600
for y in range(450, 600, 25):
    # Find sequence of classes along row y
    row = pred[y]
    # Compress consecutive identical classes
    diffs = np.where(row[:-1] != row[1:])[0]
    starts = np.r_[0, diffs + 1]
    ends = np.r_[diffs + 1, len(row)]
    seq = []
    for s, e in zip(starts, ends):
        cls_id = row[s]
        cls_name = seg_model.config.id2label[cls_id]
        seq.append(f"{cls_name}({s}-{e})")
    print(f"y={y:3d}: {' -> '.join(seq[:8])}")
