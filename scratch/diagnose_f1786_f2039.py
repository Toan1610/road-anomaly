import cv2
import numpy as np
import torch
import sys, os

# Setup path for PIDNet
PIDNET_ROOT = os.path.abspath("PIDNet")
if PIDNET_ROOT not in sys.path:
    sys.path.insert(0, PIDNET_ROOT)
    sys.path.insert(0, os.path.join(PIDNET_ROOT, "tools"))

from ultralytics import YOLO
import models as pidnet_models

def load_pidnet(arch, ckpt_path):
    model = pidnet_models.pidnet.get_pred_model(arch, 19)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v for k, v in ckpt.items()
        if k[6:] in model_dict and model_dict[k[6:]].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

yolo_model = YOLO("Road_Segmentation_Model/yolo26n-sem.pt")
pidnet_model = load_pidnet("pidnet-s", "PIDNet/pretrained_models/cityscapes/PIDNet_S_Cityscapes_val.pt").to(device)

cap = cv2.VideoCapture("video/test4.mp4")

# Inspect frames: 1786 and 2039
for target_f in [1786, 2039]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f - 1)
    ret, frame_bgr = cap.read()
    h, w = frame_bgr.shape[:2]
    
    # 1. YOLO
    yolo_h = max(32, (h + 31) // 32 * 32)
    yolo_w = max(32, (w + 31) // 32 * 32)
    yolo_res = yolo_model(frame_bgr, imgsz=(yolo_h, yolo_w), device=0, verbose=False)[0]
    yolo_pred_raw = yolo_res.semantic_mask.data.cpu().numpy()
    if yolo_pred_raw.shape != (h, w):
        yolo_pred = cv2.resize(yolo_pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        yolo_pred = yolo_pred_raw
        
    # 2. PIDNet
    inp = cv2.resize(frame_bgr, (2048, 1024))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = pidnet_model(inp)
        if isinstance(out, (list, tuple)):
            out = out[1] # high-res branch
        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        pred = out.argmax(dim=1)[0].cpu().numpy()
        
    print(f"\n================ FRAME {target_f} ================")
    # Check classes in bottom-right corner: x in [1600, 1920], y in [800, 1080]
    roi_pid = pred[800:1080, 1600:1920]
    roi_yolo = yolo_pred[800:1080, 1600:1920]
    
    unique_pid, counts_pid = np.unique(roi_pid, return_counts=True)
    print("PIDNet classes in bottom-right [800:1080, 1600:1920]:")
    for u, c in zip(unique_pid, counts_pid):
        print(f"  Class {u}: {c} pixels")
        
    unique_yolo, counts_yolo = np.unique(roi_yolo, return_counts=True)
    print("YOLO classes in bottom-right [800:1080, 1600:1920]:")
    for u, c in zip(unique_yolo, counts_yolo):
        print(f"  Class {u}: {c} pixels")

cap.release()
