import cv2
import numpy as np

from calibration import LocalView
from slot_classifier import crop_polygon, extract_features


def median_stack(image_paths):
    """Median-combine every frame from one camera into a single reference image.

    Cars, people, and moving reflections differ frame to frame and get washed
    out by the median; permanent floor paint (parking lines) stays put and
    survives sharply. Needs a reasonable number of frames per camera (~dozens)
    to be effective -- a handful of frames still helps but leaves more noise.
    """
    imgs = [cv2.imread(p) for p in image_paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise ValueError("no readable images to stack")
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
    cnt = np.asarray(cnt).reshape(-1, 1, 2)
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
