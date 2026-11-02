"""Export same-frame comparisons from saved pre-fix tensors and final videos."""
from pathlib import Path
import cv2
import numpy as np
import json

ROOT=Path(__file__).resolve().parents[2]
DEST=Path(__file__).resolve().parent

def read_frame(path,f):
    cap=cv2.VideoCapture(str(path))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES,f-1)
    ok,im=cap.read();cap.release()
    if not ok:raise RuntimeError((path,f,total))
    return im,total

def tile(im,title):
    if im.ndim==2:im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    im=cv2.resize(im,(640,360))
    cv2.rectangle(im,(0,0),(640,30),(0,0,0),-1)
    cv2.putText(im,title,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1)
    return im

all_stats={}
for video,frames in [('test6',[300,669,950,974,1000]),('test2',[100,190])]:
    output=ROOT/f'output_server_m2f_repaired_{video}'
    stats={}
    for f in frames:
        saved=DEST/video/f'frame_{f:04d}.npz'
        if not saved.exists():continue
        old=np.load(saved);base=old['frame'];raw=old['pred']==0
        old_view=base.copy()
        cnts,_=cv2.findContours(raw.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(old_view,cnts,-1,(255,0,255),2)
        cnts,_=cv2.findContours(old['corridor'],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(old_view,cnts,-1,(0,255,255),2)
        overlay,total=read_frame(output/'overlay.mp4',f)
        masks={}
        for name in ['road_mask','semantic_road_mask','corridor_mask','encroach_mask','surface_uncertain']:
            im,_=read_frame(output/(name+'.mp4'),f)
            masks[name]=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)>127
        road=masks['road_mask'];corridor=masks['corridor_mask'];encroach=masks['encroach_mask']
        yy,xx=np.indices(raw.shape);region=(yy>=360)&(yy<600)
        stats[f]={
            'total_video_frames':total,
            'old_road_outside_corridor_y360_600':int((raw&(old['corridor']==0)&region).sum()),
            'new_road_outside_corridor_y360_600':int((raw&~corridor&region).sum()),
            'encroach_outside_corridor_px':int((encroach&~corridor).sum()),
            'semantic_road_changed_vs_pre_fix_px':int((masks['semantic_road_mask']!=raw).sum()),
        }
        if video=='test6' and f==974:
            box=(slice(470,600),slice(30,180))
            stats[f]['sand_context_box_xyxy']=[30,470,180,600]
            stats[f]['old_road_pct_in_sand_context']=float(raw[box].mean()*100)
            stats[f]['new_road_pct_in_sand_context']=float(road[box].mean()*100)
        sheet=np.vstack([
            np.hstack([tile(base,f'{video} frame {f}: source'),tile(old_view,'Before: yellow=corridor magenta=semantic road'),tile(overlay,'After: yellow=visible cyan=reconstructed red=detected')]),
            np.hstack([tile(raw.astype(np.uint8)*255,'Original semantic road'),tile(road.astype(np.uint8)*255,'Refined visible road'),tile(encroach.astype(np.uint8)*255,'Final encroachment mask (no drawn boundary)')])])
        cv2.imwrite(str(DEST/f'comparison_{video}_{f}.jpg'),sheet)
    all_stats[video]=stats
(DEST/'fix_validation.json').write_text(json.dumps(all_stats,indent=2))
print(json.dumps(all_stats,indent=2))
