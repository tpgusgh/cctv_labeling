"""Row-adjacency rescue: merge the "slots sit in contiguous rows" prior with
the trained model's sub-threshold detections.

A weak model detection (below the normal confidence cutoff) is usually
noise, and a geometric row-gap is usually a pillar -- both signals alone
were measured useless (leave-one-out gap-filling: ~3% recovery; raw weak
detections: mostly junk). But TOGETHER they work: a weak detection whose
polygon sits side-by-side with an already-confirmed slot (same radius from
the fisheye center, within ~2 slot-widths in angle) is almost always a real
slot the model undersold. Measured on all 23 production cameras: 5 real
slots rescued (conf 0.07-0.21), 0 rescued detections on known-junk regions.

Isolated weak detections stay dropped; pillar gaps stay empty (the model
sees nothing there to rescue).
"""
import numpy as np

MAX_RADIUS_DIFF_FRACTION = 0.22  # same row = similar radius from the center
MAX_NEIGHBOR_WIDTHS = 2.2        # angular distance budget, in slot widths


def _polar(polygon, cx, cy):
    pts = np.asarray(polygon, dtype=np.float64)
    c = pts.mean(axis=0)
    r = float(np.hypot(c[0] - cx, c[1] - cy))
    theta = float(np.arctan2(c[1] - cy, c[0] - cx))
    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    ang_width = min(edges) / max(r, 1e-6)
    return r, theta, ang_width


def _row_adjacent(weak_polygon, strong_polygons, cx, cy):
    rw, tw, aw = _polar(weak_polygon, cx, cy)
    for sp in strong_polygons:
        rs, ts, a_s = _polar(sp, cx, cy)
        if abs(rw - rs) > MAX_RADIUS_DIFF_FRACTION * max(rw, rs):
            continue
        dtheta = abs(np.arctan2(np.sin(tw - ts), np.cos(tw - ts)))
        if dtheta < MAX_NEIGHBOR_WIDTHS * max(aw, a_s):
            return True
    return False


def rescue_row_adjacent(weak_detections, strong_detections, center):
    """weak_detections: model output below the normal confidence cutoff,
    already deduplicated against strong_detections by the caller.
    Returns the subset worth keeping, tagged with source='row-rescue'."""
    cx, cy = float(center[0]), float(center[1])
    strong_polygons = [d["polygon"] for d in strong_detections]
    if not strong_polygons:
        return []
    rescued = []
    for d in weak_detections:
        if _row_adjacent(d["polygon"], strong_polygons, cx, cy):
            rescued.append(dict(d, source="row-rescue"))
    return rescued
