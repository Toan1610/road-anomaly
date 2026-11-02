import cv2
import torch
import numpy as np
from run_video_encroachment_mask2former import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/mask2former-swin-tiny-cityscapes-semantic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name, local_files_only=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, local_files_only=True).to(device)
model.eval()

cap = cv2.VideoCapture("video/test2.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 189)
ret, frame = cap.read()
H, W = frame.shape[:2]
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

probs = infer_mask2former_probs(model, processor, frame_rgb, H, W, device, True)
max_prob, pred_tensor = torch.max(probs, dim=0)

crop_prob = max_prob[480:680, 0:300].cpu().numpy()
crop_pred = pred_tensor[480:680, 0:300].cpu().numpy()

print(f"Max prob in sand pile: min={crop_prob.min():.3f}, mean={crop_prob.mean():.3f}, max={crop_prob.max():.3f}")
print(f"Pixels with max_prob < 0.35: {np.count_nonzero(crop_prob < 0.35)} / {crop_prob.size} ({np.count_nonzero(crop_prob < 0.35)/crop_prob.size*100:.1f}%)")

