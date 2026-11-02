import numpy as np

def match_score(track, det, frame_idx, fps=24.0):
    # Check side
    if track.get("side") != det.get("side"):
        return 0.0
        
    t_bx, t_by, t_bw, t_bh = track["bbox"]
    d_bx, d_by, d_bw, d_bh = det["bbox"]
    
    # 1. Standard IoU
    xA = max(t_bx, d_bx)
    yA = max(t_by, d_by)
    xB = min(t_bx + t_bw, d_bx + d_bw)
    yB = min(t_by + t_bh, d_by + d_bh)
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (t_bw * t_bh) + (d_bw * d_bh) - inter
    iou = inter / max(1.0, union)
    if iou > 0.10:
        return 1.0 + iou

    # 2. Ego-motion forward projection
    # Camera moving forward -> obstacles move downward (y increases) and outward to the left (x decreases)
    dt_frames = frame_idx - track["last_seen"]
    dt_sec = dt_frames / fps
    if dt_sec > 2.0:
        return 0.0

    # Centers
    tc_x = t_bx + t_bw * 0.5
    tc_y = t_by + t_bh * 0.5
    dc_x = d_bx + d_bw * 0.5
    dc_y = d_by + d_bh * 0.5

    # As car moves forward, y should increase (object gets closer) or be nearby
    dy = dc_y - tc_y
    dx = dc_x - tc_x

    # For left-side obstacle: dx is typically <= +50 (drifts left as we pass it), dy is >= -20
    # Center distance normalized by frame dimensions (1280x720)
    dist_norm = np.sqrt((dx / 1280.0)**2 + (dy / 720.0)**2)
    
    # If within reasonable distance for the elapsed time
    max_allowed_dist = 0.10 + 0.15 * dt_sec  # ~120px to 300px
    if dist_norm < max_allowed_dist and dy >= -30:
        score = 1.0 - (dist_norm / max_allowed_dist)
        return score
    return 0.0

print("match_score defined.")
