import os
os.environ['HF_HUB_OFFLINE']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

out=Path('scratch/audit_test6/resolution'); out.mkdir(exist_ok=True)
model=Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic',local_files_only=True).cuda().eval()
mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
for video,f in [('test6',974),('test6',950),('test2',190),('test6',300)]:
    d=np.load(f'scratch/audit_test6/{video}/frame_{f:04d}.npz'); bgr=d['frame']; h,w=bgr.shape[:2]
    rows=[]
    for ih,iw in [(384,384),(512,928),(720,1280)]:
        t=torch.from_numpy(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).cuda().permute(2,0,1)[None].float()/255
        t=(F.interpolate(t,size=(ih,iw),mode='bilinear',align_corners=False)-mean)/std
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            o=model(pixel_values=t)
            classes=o.class_queries_logits.float().softmax(-1)[...,:-1]
            masks=F.interpolate(o.masks_queries_logits.float(),size=(ih,iw),mode='bilinear',align_corners=False).sigmoid()
            scores=torch.einsum('bqc,bqhw->bchw',classes,masks)
            scores=F.interpolate(scores,size=(h,w),mode='bilinear',align_corners=False)[0]
        p=scores.argmax(0).cpu().numpy().astype(np.uint8)
        s=scores.max(0).values.cpu().numpy()
        np.savez_compressed(out/f'{video}_{f}_{ih}.npz',pred=p,scores=s,frame=bgr)
        im=bgr.copy(); im[p==0]=(im[p==0]*.5+np.array([0,125,0])).astype(np.uint8)
        cv2.putText(im,f'{video} f{f}: {iw}x{ih}',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
        rows.append(cv2.resize(im,(640,360)))
        print(video,f,ih,'road',int((p==0).sum()),flush=True)
    cv2.imwrite(str(out/f'{video}_{f}.jpg'),np.vstack(rows))
