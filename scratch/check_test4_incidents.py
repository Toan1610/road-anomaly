import cv2
import json
import numpy as np

with open("output_test4_pidnet/incidents/incidents_log.json") as f:
    incidents = json.load(f)

for inc in incidents:
    print(f"Incident {inc['incident_id']}: peak_frame={inc['peak_frame']}, time={inc['timestamp_sec']}s, dur={inc['duration_sec']}s ({inc['duration_frames']} frames), side={inc['side']}, depth={inc['max_depth_px']}, area={inc['max_area_px']}, bbox={inc['bounding_box_xywh']}")
