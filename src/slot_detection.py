import cv2
import numpy as np

from calibration import LocalView
from slot_classifier import crop_polygon, extract_features


MAX_MEDIAN_FRAMES = 60


def _polygon_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_centroid_and_diagonal(bbox):
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2), ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5


def _same_slot(bbox_a, bbox_b, iou_threshold):
    if _bbox_iou(bbox_a, bbox_b) > iou_threshold:
        return True
    # Two detectors can fit visibly different exact quads to the same
    # physical slot (different corner rounding/edge snapping) -- enough that
    # bbox IoU alone stays under iou_threshold despite it being one slot.
    # Verified against production configs: same slot detected twice with
    # centroids a few px apart but bbox IoU as low as 0.3, both surviving
    # this dedup and only becoming visible once the rendered label box grew
    # large enough to visibly overlap. Centroid proximity relative to box
    # size catches that specific case without loosening iou_threshold itself
    # (which would risk merging genuinely distinct adjacent slots).
    centroid_a, diag_a = _bbox_centroid_and_diagonal(bbox_a)
    centroid_b, diag_b = _bbox_centroid_and_diagonal(bbox_b)
    dist = ((centroid_a[0] - centroid_b[0]) ** 2 + (centroid_a[1] - centroid_b[1]) ** 2) ** 0.5
    avg_diag = (diag_a + diag_b) / 2
    return avg_diag > 0 and dist / avg_diag < 0.15


def regularize_quad(polygon, calibration, patch_size=(300, 300), local_f=300.0, min_aspect=1.6):
    """Snap a detected quad to a physically plausible slot shape using the
    slot's own locally-rectified (distortion-free) view.

    Two verified failure modes of fitting quads directly in raw fisheye
    space (user report: '누가봐도 일직선인데 대각선', edge slots '납작해짐'):
    - fisheye curvature skews a straight slot's raw contour, so the fitted
      quad comes out diagonal even though the slot is straight;
    - near the image edge the mask loses the far end of the bay, so the quad
      is far too shallow (flat) compared to a real ~2.3x5m slot.

    In the local gnomonic view a real slot IS a rectangle, and in this
    facility every bay's depth axis points at the camera (ring layout, the
    user's description: edge slots stand up '벽처럼'). So:
    1. project the 4 corners into the local view;
    2. if the detected shape's principal axis is within ~35 deg of the
       radial direction, rebuild the rectangle ON the radial axes (kills
       residual diagonal skew on straight slots);
    3. if the rectangle is flatter than a real bay (depth/width below
       min_aspect), extend the far (outward) edge -- with hard caps, since
       a previous uncapped version blew up near the equidistant model's
       tan() singularity at the image edge (runaway quads);
    4. project back to raw and sanity-check; fall back to the plain
       min-area-rect snap, then to the original polygon, if anything fails."""
    try:
        pts = np.asarray(polygon, dtype=np.float64)
        centroid = pts.mean(axis=0)
        view = LocalView.centered_on(calibration, tuple(centroid), patch_size, local_f)
        local_pts = np.asarray(view.raw_to_local(pts), dtype=np.float64)
        if not np.all(np.isfinite(local_pts)):
            return polygon
        rect = cv2.minAreaRect(local_pts.astype(np.float32))
        snap_box = cv2.boxPoints(rect).astype(np.float64)

        def unproject(box):
            out = np.asarray(view.local_to_raw(box))
            if not np.all(np.isfinite(out)):
                return None
            orig_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            new_diag = np.linalg.norm(out.max(axis=0) - out.min(axis=0))
            if orig_diag > 1e-6 and new_diag > 3.0 * orig_diag:
                return None  # runaway near the fisheye edge -- reject
            return out.tolist()

        # radial (away-from-camera-center) direction in local space
        radial = centroid - np.array([calibration.cx, calibration.cy])
        rn = np.linalg.norm(radial)
        if rn < 1e-6:
            return unproject(snap_box) or polygon
        two = np.asarray(view.raw_to_local(np.stack([centroid, centroid + 20.0 * radial / rn])), dtype=np.float64)
        d_axis = two[1] - two[0]
        dn = np.linalg.norm(d_axis)
        if dn < 1e-6:
            return unproject(snap_box) or polygon
        d_axis /= dn

        # KEEP the detection's own orientation (forcing the depth axis onto
        # the exact radial direction was tried and reverted: bay rows here
        # are straight segments, so end-of-row bays deviate from radial and
        # the forced rotation made straight slots look diagonal -- the exact
        # problem it was meant to fix). Radial is only used to pick which
        # rect axis is depth and which way is outward, for flat-quad extension.
        e0 = snap_box[1] - snap_box[0]
        e1 = snap_box[2] - snap_box[1]
        l0, l1 = np.linalg.norm(e0), np.linalg.norm(e1)
        if min(l0, l1) < 1e-6:
            return polygon
        u0, u1 = e0 / l0, e1 / l1
        if abs(np.dot(u0, d_axis)) >= abs(np.dot(u1, d_axis)):
            depth_axis, width_axis = u0, u1
        else:
            depth_axis, width_axis = u1, u0
        if np.dot(depth_axis, d_axis) < 0:
            depth_axis = -depth_axis

        center = local_pts.mean(axis=0)
        pd = (local_pts - center) @ depth_axis
        pw = (local_pts - center) @ width_axis
        dmin, dmax = pd.min(), pd.max()
        wmin, wmax = pw.min(), pw.max()
        width_len = wmax - wmin
        if width_len < 1e-6:
            return unproject(snap_box) or polygon

        def build(dmax_v):
            return np.array([
                center + dmin * depth_axis + wmin * width_axis,
                center + dmin * depth_axis + wmax * width_axis,
                center + dmax_v * depth_axis + wmax * width_axis,
                center + dmax_v * depth_axis + wmin * width_axis,
            ])

        def raw_aspect(raw_box):
            rb = np.asarray(raw_box, dtype=np.float64)
            c = rb.mean(axis=0)
            rd = c - np.array([calibration.cx, calibration.cy])
            rd_n = np.linalg.norm(rd)
            if rd_n < 1e-6:
                return None
            rd /= rd_n
            wd = np.array([-rd[1], rd[0]])
            p_d = (rb - c) @ rd
            p_w = (rb - c) @ wd
            w_ext = p_w.max() - p_w.min()
            if w_ext < 1e-6:
                return None
            return (p_d.max() - p_d.min()) / w_ext

        # enforce the slot's proportions in RAW image space, not the local
        # plane: near the fisheye edge the unprojection compresses the
        # radial direction again, so a locally-1.6 rectangle still lands
        # flat on screen ('멀어질수록 납작'). Extend the far (outward) end
        # step by step until the on-screen depth/width ratio looks like a
        # real standing bay -- hard-capped, and any step whose unprojection
        # blows up (fisheye-edge singularity) rolls back to the last good box.
        best = unproject(build(dmax))
        max_extension = min(2.0 * width_len, 120.0)
        step = max(width_len * 0.15, 4.0)
        extended = 0.0
        while extended < max_extension:
            cur = raw_aspect(best) if best is not None else None
            if best is not None and cur is not None and cur >= min_aspect:
                break
            candidate = unproject(build(dmax + extended + step))
            if candidate is None:
                break
            best = candidate
            extended += step
        return best or unproject(snap_box) or polygon
    except Exception:
        return polygon


def is_degenerate_quad(polygon, min_area_ratio=0.6, min_short_side=9.0, max_aspect=8.0):
    """True for quads that cannot be a real parking slot outline: bowtie /
    self-intersecting (shoelace area collapses vs the min-area rect), folded
    shapes, or extreme slivers. Real fisheye-edge slots ARE thin and tilted
    (verified real slots down to ~16px short side, aspect ~5.7), so the
    thresholds only cut clearly-broken geometry -- the user's report: tilted
    slots are correct, '납작해진' (squashed/folded) ones are junk and must
    neither ship in configs nor become training data."""
    pts = np.asarray(polygon, dtype=np.float32)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return True
    # shoelace area: a bowtie's signed halves cancel, so this collapses
    x, y = pts[:, 0], pts[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    (rw, rh) = cv2.minAreaRect(pts)[1]
    rect_area = rw * rh
    if rect_area <= 0 or shoelace <= 0:
        return True
    short, long_ = min(rw, rh), max(rw, rh)
    if shoelace / rect_area < min_area_ratio:
        return True
    if short < min_short_side:
        return True
    if long_ / max(short, 1e-6) > max_aspect:
        return True
    return False


def merge_detections(*detection_lists, iou_threshold=0.4):
    """Union multiple independent detectors' candidate lists into one,
    deduping overlapping detections (the same real slot found by more than
    one detector) by keeping the higher-confidence one.

    Raises overall recall for free -- a slot the trained model missed but
    classical CV (or vice versa) still caught makes it into the candidate
    list -- without needing more training data, since a human reviews every
    candidate either way (see generate_config.py). Detectors disagreeing
    isn't treated as a problem to resolve here; the review queue is exactly
    where that gets sorted out.

    Each returned detection also gets an "agreement_count" key: how many of
    the *distinct input lists* had an overlapping candidate for it (1 = only
    one detector found it, 2+ = independent detectors agree). Two
    differently-erroring detectors independently landing on the same region
    is a real quality signal -- generate_config.py uses it to optionally
    auto-accept the highest-confidence tier of candidates, cutting down how
    much a human has to click through without needing more labeled data.
    """
    tagged = [(d, list_idx) for list_idx, lst in enumerate(detection_lists) for d in lst]
    tagged.sort(key=lambda t: t[0]["confidence"], reverse=True)

    kept = []
    kept_bboxes = []
    kept_sources = []
    for d, list_idx in tagged:
        bbox = _polygon_bbox(d["polygon"])
        match = next((i for i, kb in enumerate(kept_bboxes) if _same_slot(bbox, kb, iou_threshold)), None)
        if match is not None:
            kept_sources[match].add(list_idx)
            continue
        kept.append(dict(d))
        kept_bboxes.append(bbox)
        kept_sources.append({list_idx})

    for d, sources in zip(kept, kept_sources):
        d["agreement_count"] = len(sources)
    return kept


def median_stack(image_paths):
    """Median-combine every frame from one camera into a single reference image.

    Cars, people, and moving reflections differ frame to frame and get washed
    out by the median; permanent floor paint (parking lines) stays put and
    survives sharply. Needs a reasonable number of frames per camera (~dozens)
    to be effective -- a handful of frames still helps but leaves more noise.

    ponytail: caps at MAX_MEDIAN_FRAMES (evenly subsampled, not just the
    first N) instead of loading every frame as a simultaneous float32 array
    -- that's N_frames * H * W * 3 * 4 bytes, which is fine at this
    project's native 640x640 but can OOM-kill the whole process on
    full-resolution phone photos (e.g. 4000x3000) with dozens+ frames in
    one camera folder. Upgrade path if 60 stops being enough signal: a
    true streaming/chunked median instead of subsampling.
    """
    if len(image_paths) > MAX_MEDIAN_FRAMES:
        step = len(image_paths) / MAX_MEDIAN_FRAMES
        image_paths = [image_paths[int(i * step)] for i in range(MAX_MEDIAN_FRAMES)]
    imgs = [cv2.imread(p) for p in image_paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise ValueError("no readable images to stack")
    # a real folder drag-in can mix resolutions (unrelated photos grouped
    # under one folder name) -- np.stack would crash the whole job on shape
    # mismatch. Keep the majority resolution; the rest can't be frames of
    # the same fixed camera anyway.
    shapes = {}
    for im in imgs:
        shapes.setdefault(im.shape, []).append(im)
    imgs = max(shapes.values(), key=len)
    stack = np.stack(imgs, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def _build_edge_mask(median_bgr, calibration, patch_size=(220, 220), local_f=220.0, floor_radius=300, grid_step=80):
    # ponytail: raw Canny/Hough on the fisheye frame misses any real slot line
    # that doesn't pass through the lens's optical center, since equidistant
    # fisheye projection bends those into curves. Rectifying many small
    # overlapping local patches (LocalView, already built for per-slot label
    # rendering) keeps each patch close to distortion-free, so Hough finds the
    # true line, then we map the detected segment back to raw pixel space.
    h, w = median_bgr.shape[:2]
    centers = []
    for gx in range(grid_step, w - grid_step + 1, grid_step):
        for gy in range(grid_step, h - grid_step + 1, grid_step):
            if (gx - calibration.cx) ** 2 + (gy - calibration.cy) ** 2 <= (floor_radius - 30) ** 2:
                centers.append((gx, gy))

    edge_mask = np.zeros((h, w), np.uint8)
    for center in centers:
        view = LocalView.centered_on(calibration, center, patch_size, local_f)
        patch = view.rectify(median_bgr)
        # ponytail: grayscale-only Canny misses a line whose paint differs in
        # hue but not in luminance from the floor (some slots use colored
        # lines, not just white). Saturation channel edges catch what
        # brightness alone can't; union of both channels beats either alone.
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 110)
        sat = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
        sat_edges = cv2.Canny(cv2.GaussianBlur(sat, (3, 3), 0), 35, 110)
        edges = cv2.bitwise_or(gray_edges, sat_edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=35, maxLineGap=6)
        if lines is None:
            continue
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4).astype(float)
            local_pts = np.stack([np.linspace(x1, x2, 40), np.linspace(y1, y2, 40)], axis=1)
            raw_pts = view.local_to_raw(local_pts)
            for px, py in raw_pts:
                pxi, pyi = int(round(px)), int(round(py))
                if 0 <= pxi < w and 0 <= pyi < h:
                    cv2.circle(edge_mask, (pxi, pyi), 1, 255, -1)

    circle_mask = np.zeros((h, w), np.uint8)
    cv2.circle(circle_mask, (int(calibration.cx), int(calibration.cy)), floor_radius, 255, -1)
    return cv2.bitwise_and(edge_mask, edge_mask, mask=circle_mask), circle_mask


def _is_hazard_striped(bgr, comp_mask, sat_thresh=50, hue_lo=12, hue_hi=45, frac_thresh=0.12):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    region = hsv[comp_mask > 0]
    if len(region) == 0:
        return False
    yellowish = (region[:, 0] >= hue_lo) & (region[:, 0] <= hue_hi) & (region[:, 1] >= sat_thresh)
    return bool(yellowish.mean() > frac_thresh)


def _shrink_polygon(poly, inset_px=6.0):
    # ponytail: morphological closing (in _build_edge_mask's caller) grows the
    # detected boundary a few px past the real paint line. Pull each corner
    # inward toward the polygon centroid so the rendered outline stays inside
    # the true white line instead of overshooting it.
    poly = np.asarray(poly, dtype=np.float64)
    centroid = poly.mean(axis=0)
    shrunk = []
    for p in poly:
        v = centroid - p
        dist = np.linalg.norm(v)
        if dist < 1e-6:
            shrunk.append(p)
            continue
        move = min(inset_px, dist * 0.4)
        shrunk.append(p + v / dist * move)
    return np.array(shrunk)


def fit_quad(cnt, inset_px=6.0):
    """Reduce an arbitrary contour/polygon to a 4-point quad.

    perspective.plane_to_pixel_homography requires exactly 4 points, but a
    real contour (classical CV connected-component boundary, or a YOLO-seg
    mask polygon) rarely comes out as a clean quad. Try a 4-corner
    approximation first; if the shape doesn't reduce to 4 corners, fall back
    to its minimum-area bounding rect. Either way, shrink the result inward
    so it doesn't overshoot the true painted line (see _shrink_polygon).
    """
    cnt = np.asarray(cnt)
    if cnt.dtype not in (np.int32, np.float32):
        cnt = cnt.astype(np.float32)
    cnt = cnt.reshape(-1, 1, 2)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        poly = approx.reshape(-1, 2)
    else:
        rect = cv2.minAreaRect(cnt)
        poly = cv2.boxPoints(rect)
    return _shrink_polygon(poly, inset_px)


def detect_slots(median_bgr, calibration, min_area=2800, max_area=9500, min_rectangularity=0.65, inset_px=6.0,
                  floor_radius=300, classifier=None):
    """Find candidate parking-slot polygons in a camera's median-stacked reference image.

    Returns a list of {"polygon": [[x, y] x4], "confidence": float}, confidence
    being the fitted quad's fill ratio (rectangularity) against its own
    contour area -- a cheap, unitless quality signal, not a trained score.

    classifier: optional trained model (see slot_classifier.train) fit on
    human accept/reject review feedback. When given, any candidate that
    passes every geometric/color filter below is additionally cropped and
    run through the classifier; a "reject" prediction drops it from the
    results. Default None keeps existing behavior unchanged (opt-in, same
    pattern as the old require_studs parameter).

    ponytail: min_area/max_area are pixel-unit thresholds tuned for this
    project's f=204/radius=320 camera model. Known false-positive class: a
    real floor marking that isn't a parking slot (lane arrows, warning signs,
    crosswalk gaps, background structure) can still pass every geometric/
    color filter here, since telling those apart from a slot is a content
    question, not a shape one -- route low-confidence/low-count cameras to
    manual review rather than trusting this blindly (see generate_config.py).
    A wheel-stop-stud confirmation pass was tried (small dark dot cluster on
    the entrance edge, visually confirmed real on every tested slot and
    absent on every tested false positive) but 5 parameterizations across two
    techniques (brightness-threshold blob counting, HoughCircles) couldn't
    separate real studs from noise at this camera's resolution (~2-3px
    studs) -- see HANDOFF.md before re-attempting that angle.
    """
    edge_mask, circle_mask = _build_edge_mask(median_bgr, calibration, floor_radius=floor_radius)
    closed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    floor_cells = cv2.bitwise_and(cv2.bitwise_not(closed), circle_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(floor_cells, connectivity=4)

    results = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area or area > max_area:
            continue
        comp_mask = (labels == i).astype(np.uint8) * 255
        if _is_hazard_striped(median_bgr, comp_mask):
            continue
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < min_rectangularity:
            continue
        poly = fit_quad(cnt, inset_px)
        confidence = round(float(min(rectangularity, 1.0)), 3)

        if classifier is not None:
            crop = crop_polygon(median_bgr, poly)
            if crop.size == 0 or classifier.predict([extract_features(crop, confidence)])[0] == 0:
                continue

        results.append({"polygon": poly.tolist(), "confidence": confidence})
    return results
