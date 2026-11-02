"""Observed road edges, bounded occlusion reconstruction and surface uncertainty.

No camera-specific perspective template is used. Semantic uncertainty is an
auxiliary cue, not an unknown-object class or a calibrated OOD probability.
"""
import cv2
import numpy as np


def largest_road_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def bottom_foreground_mask(pred, road_id=0):
    """Conservative automatic camera-foreground ROI; custom polygons can override it."""
    h, w = pred.shape
    top = int(h * .65)
    foreground = (pred[top:] != road_id).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    out = np.zeros_like(pred, dtype=np.uint8)
    for label in np.unique(labels[-1]):
        if label == 0:
            continue
        x, y, bw, bh, area = stats[label]
        # Small objects and narrow foreground objects touching the bottom remain visible.
        if area >= h * w * .012 and bw >= w * .20:
            out[top:][labels == label] = 255
    return cv2.dilate(out, np.ones((5, 5), np.uint8))


def camera_foreground_neighborhood(calibrated):
    """Compute once after startup calibration; large ellipse dilation is expensive."""
    core=calibrated.copy()
    core[int(core.shape[0]*.92):]=0
    yy,xx=np.where(core>0)
    if not len(xx) or np.ptp(xx) > calibrated.shape[1]*.75:
        # A thin dashboard spanning the image must not grow into an approaching pile.
        return np.zeros_like(calibrated,dtype=bool)
    h,w=calibrated.shape
    kx=max(3,int(w*.24)//2*2+1)
    ky=max(3,int(h*.24)//2*2+1)
    return cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kx,ky)))>0


def expand_camera_foreground(calibrated, pred, road_id=0, neighborhood=None):
    """Follow foreground motion near a calibrated camera part, not new distant objects."""
    near=camera_foreground_neighborhood(calibrated) if neighborhood is None else neighborhood
    current=bottom_foreground_mask(pred,road_id)>0
    # Segmentation mixes camera and road pixels along the foreground edge.
    radius=max(3,int(pred.shape[1]*.018))
    current=cv2.dilate(current.astype(np.uint8),np.ones((radius*2+1,radius*2+1),np.uint8))>0
    return ((calibrated>0)|(near & current)).astype(np.uint8)*255


def has_obstacle_support(component, surface_core, enclosed_nonroad):
    """Reject partial-volume curb bands without interior object evidence."""
    size=int(component.sum())
    if not size:
        return False
    core=int(np.count_nonzero(component & surface_core))
    holes=int(np.count_nonzero(component & enclosed_nonroad))
    return core >= max(50,size*.03) or holes >= max(200,size*.25)


def uncertain_road_surface(frame_bgr, pred, scores, road_id=0, exempt=None):
    """Grow weak road support only from weak non-road material seeds.

    Chroma agreement with the adjacent seed prevents an uncertainty band from
    indiscriminately consuming asphalt. Confident road is never relabelled.
    This intentionally cannot recover material confidently labelled as road.
    """
    road = pred == road_id
    out = np.zeros(pred.shape, np.uint8)
    if scores is None:
        return out
    blocked = np.zeros_like(road) if exempt is None else exempt > 0
    blocked = cv2.dilate(blocked.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    weak = (scores < .60) & ~blocked
    seed = (pred != road_id) & (scores < .10) & ~blocked
    # Work at half resolution; retain full-resolution score gates below.
    small = cv2.resize(weak.astype(np.uint8), None, fx=.5, fy=.5, interpolation=cv2.INTER_NEAREST)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(small, 8)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    full_labels = cv2.resize(labels, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 100:
            continue
        x, y, bw, bh = stats[label, :4] * 2
        y2, x2 = min(pred.shape[0], y+bh+2), min(pred.shape[1], x+bw+2)
        component = full_labels[y:y2, x:x2] == label
        local_seed = seed[y:y2,x:x2] & component
        if local_seed.sum() < 100:
            continue
        ab = lab[y:y2,x:x2,1:].astype(np.float32)
        seed_color = np.median(ab[local_seed],axis=0)
        agrees = np.linalg.norm(ab - seed_color,axis=2) < 10.
        local = component & road[y:y2,x:x2] & weak[y:y2,x:x2] & agrees
        out[y:y2,x:x2][local] = 255
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))


def reconstruct_corridor(mask_road, occluders, y_top, y_bot,
                         prev_left=None, prev_right=None, reference=None,
                         diagnostics=None):
    """Preserve measured edges; interpolate occlusions using measured support.

    A gap needs measured support on both sides. An unbounded gap may use a
    nearby outward warped prior; otherwise retain the measured road extent.
    Rows without road support stay empty. No perspective template is used.
    """
    h,w = mask_road.shape
    road = largest_road_component(mask_road)
    ys = np.arange(y_top,y_bot+1)
    has = road[ys].any(axis=1)
    left = np.argmax(road[ys],axis=1).astype(np.float32)
    right = (w-1-np.argmax(road[ys,::-1],axis=1)).astype(np.float32)
    obstacle = cv2.dilate((occluders > 0).astype(np.uint8), np.ones((9,9),np.uint8)) > 0
    outputs=[]
    observed_flags=[]
    for side, edge, previous in [(0,left,prev_left),(1,right,prev_right)]:
        observed = has & ~obstacle[ys,np.clip(edge.astype(int),0,w-1)]
        idx = np.flatnonzero(observed)
        curve = edge.copy()
        inferred = ~observed
        if idx.size >= 2:
            bounded = inferred & has & (ys > ys[idx[0]]) & (ys < ys[idx[-1]])
            curve[bounded] = np.interp(ys[bounded],ys[idx],edge[idx])
        # A one-sided gap has no second anchor. Do not invent a straight curb
        # across the whole image. A nearby, outward prior can support it.
        if previous is not None and len(previous)==len(ys):
            prior=np.asarray(previous)
            outward=(prior<=edge) if side==0 else (prior>=edge)
            supported=inferred & has & outward & (np.abs(prior-edge)<w*.12)
            curve[supported]=prior[supported]
        if reference is not None:
            a,b=reference[side]
            estimate=a*ys+b
            supported=inferred & has & (np.abs(estimate-edge)<w*.12)
            curve[supported]=estimate[supported]
        if has.any():
            curve[~has]=np.interp(ys[~has],ys[has],curve[has])
        else:
            curve[:]=0 if side==0 else w-1
        curve=np.clip(curve,0,w-1)
        # Avoid broad Gaussian smoothing, which moves perfectly visible curbs.
        outputs.append(curve.astype(np.float32))
        observed_flags.append(observed)
    left,right=outputs
    invalid=left>right
    left[invalid]=0;right[invalid]=w-1
    corridor=np.zeros((h,w),np.uint8)
    for j,y in enumerate(ys):
        if has[j]:
            corridor[y,int(left[j]):int(right[j])+1]=255
    lp=np.column_stack((np.rint(left[has]).astype(np.int32),ys[has]))
    rp=np.column_stack((np.rint(right[has]).astype(np.int32),ys[has]))
    full_l=np.interp(np.arange(h),ys,left).astype(np.float32)
    full_r=np.interp(np.arange(h),ys,right).astype(np.float32)
    if diagnostics is not None:
        diagnostics.update(observed_left=observed_flags[0],observed_right=observed_flags[1],
                           y=ys,road=road)
    return corridor,lp,rp,full_l,full_r,left,right
