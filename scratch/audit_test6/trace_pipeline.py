"""Read-only pipeline audit: run original main and capture intermediate masks.

Run from repository root with venv/bin/python scratch/audit_test6/trace_pipeline.py.
No production source or previous output is modified. Line hooks are resolved from
source text so the audit fails explicitly if the expected statements disappear.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_server_mask2former as pipeline

video = sys.argv[1] if len(sys.argv) > 1 else 'test6'
DEST = ROOT / 'scratch' / 'audit_test6' / video
DEST.mkdir(parents=True, exist_ok=True)
source_path = Path(pipeline.__file__).resolve()
source = source_path.read_text()
lines = source.splitlines()
def locate(text):
    matches = [i + 1 for i, line in enumerate(lines) if text in line]
    assert len(matches) == 1, (text, matches)
    return matches[0]

snapshot_line = locate('contours, _ = cv2.findContours(')
candidate_line = locate('if len(candidate_detections) > 1:')
tracking_line = locate('is_alarm_active = len(active_tracks) > 0 and corridor_state == "normal"')
rejects = {n: reason for text, reason in [
    ('if area < dyn_min_area:', 'area'),
    ('if is_corner_artifact(cnt,', 'corner'),
    ('if by <= y_top + 5 and bh <= 45:', 'top'),
    ('if overlap_vehicle / cnt_px > 0.40:', 'vehicle_overlap'),
    ('if road_contact < 30:', 'road_contact'),
    ('if seg_infra_px / cnt_px > 0.40:', 'infrastructure_label'),
    ('if seg_left_terrain / cnt_px > 0.40:', 'left_terrain_label'),
    ('if (bx - curb_x) > 120:', 'left_curb'),
    ('if (curb_x - (bx + bw)) > 120:', 'right_curb'),
    ('if cnt_max_depth < dyn_min_depth:', 'depth'),
] for n in [locate(text) + 1]}
samples = {5, 150, 300, 600, 860, 900, 930, 950, 974, 1000, 1030, 1123} if video == 'test6' else {100, 150, 170, 190, 210}
stats = {}
final_incidents = []
palette = np.random.default_rng(3).integers(60, 240, (256, 3), dtype=np.uint8)
palette[0] = (40, 200, 40)
palette[1] = (0, 170, 255)
palette[9] = (160, 80, 200)

def tile(im, label):
    if im.ndim == 2:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    im = cv2.resize(im, (640, 360))
    cv2.rectangle(im, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(im, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return im

def trace(frame, event, arg):
    if frame.f_code is not pipeline.main.__code__:
        return None
    if event == 'return':
        final_incidents.extend(frame.f_locals.get('incidents_log', []))
        return trace
    if event != 'line':
        return trace
    d = frame.f_locals
    f = d.get('frame_idx', -1)
    if f not in samples:
        return trace
    lineno = frame.f_lineno
    if lineno == snapshot_line and f not in stats:
        pred = d['pred_u8']
        corridor = d['legal_corridor_poly'] > 0
        road = d['mask_road']
        h, w = pred.shape
        ys, xs = np.indices(pred.shape)
        # Rectangles are contextual samples, NOT hand-labelled object masks.
        rois = {'left_material_context': (0, 340, 560, 640),
                'right_hood_context': (700, 600, 1200, 706)} if video == 'test6' else {'left_material_context': (0, 400, 380, 650)}
        regions = {}
        for name, (x1,y1,x2,y2) in rois.items():
            roi = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
            ids, counts = np.unique(pred[roi], return_counts=True)
            regions[name] = {'bbox_xyxy': [x1,y1,x2,y2], 'labels': {pipeline.CITYSCAPES_19_LABELS.get(int(k),str(k)): int(v) for k,v in zip(ids,counts)},
                'corridor_px': int((corridor & roi).sum()),
                'anomaly_px': int((d['anomaly_core'] & roi).sum())}
        # Compare saved mask with semantic road only where neither vehicles nor hood should explain differences.
        region = (ys >= 360) & (ys < 600)
        extra = corridor & ~road & region
        missing = road & ~corridor & region
        stats[f] = {'corridor_extra_vs_road_y360_600': int(extra.sum()),
                    'road_outside_corridor_y360_600': int(missing.sum()),
                    'corridor_conf': float(d['corridor_conf']),
                    'regions': regions, 'rejected': []}
        base = cv2.cvtColor(d['frame_rgb'], cv2.COLOR_RGB2BGR)
        contour_view = base.copy()
        cv2.polylines(contour_view, [d['pts_left_curve'],d['pts_right_curve']], False, (0,255,255), 3)
        contours,_ = cv2.findContours(road.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_view, contours, -1, (255,0,255), 2)
        semantic = cv2.addWeighted(base, .5, palette[pred], .5, 0)
        sheet = np.vstack([
            np.hstack([tile(base, f'{video} frame {f}: source'),tile(semantic, 'Semantic: road=green sidewalk=orange terrain=purple')]),
            np.hstack([tile(contour_view, 'Yellow=corridor / magenta=semantic road contour'),tile(d['legal_corridor_poly'], 'Reconstructed corridor')]),
            np.hstack([tile(d['normal_mask'], 'Normal mask (includes sidewalk)'),tile(d['encroach_mask'], 'Anomaly candidates before rejection/tracking')])])
        cv2.imwrite(str(DEST / f'frame_{f:04d}.jpg'), sheet)
        np.savez_compressed(DEST / f'frame_{f:04d}.npz', pred=pred, scores=d['max_prob_np'], corridor=d['legal_corridor_poly'], anomaly=d['encroach_mask'], frame=base)
        print(f'[AUDIT] captured {video} frame {f}', flush=True)
    elif lineno in rejects and f in stats:
        stats[f]['rejected'].append({'reason': rejects[lineno], 'bbox': [int(d[k]) for k in ['bx','by','bw','bh']], 'area': float(d['area'])})
    elif lineno == candidate_line and f in stats:
        stats[f]['accepted_before_merge'] = [{'bbox':list(c['bbox']), 'area':c['area'], 'depth':float(c['depth'])} for c in d['candidate_detections']]
    elif lineno == tracking_line and f in stats:
        stats[f]['confirmed_track_ids'] = [t['id'] for t in d['active_tracks']]
    return trace

sys.argv = [str(source_path), '--input', f'video/{video}.mp4', '--output_dir', str(DEST/'run'), '--model', 'm2f-large', '--no-save_video', '--no-save_masks', '--no-save_incidents']
sys.settrace(trace)
try:
    pipeline.main()
finally:
    sys.settrace(None)
    (DEST/'stats.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(), 'samples':stats, 'incidents':final_incidents}, indent=2))
    print(f'[AUDIT] Total incidents across entire run: {len(final_incidents)}', flush=True)
