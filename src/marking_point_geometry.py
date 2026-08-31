import numpy as np

from calibration import LocalView

# ponytail: patch/focal length for the locally-rectified tangent plane used
# to measure/reconstruct direction and depth. Only needs to be big enough
# to hold one slot's corners (entrance width + depth, both well under
# 200px in this project's data) -- not an actual rendered image, so no
# accuracy cost to sizing it generously.
LOCAL_PATCH_SIZE = (400, 400)
LOCAL_FOCAL_PX = 200.0


def _rotate90_cw(vec):
    x, y = vec
    return np.array([y, -x])


def _local_view_at(calibration, points):
    center = np.asarray(points, dtype=np.float64).mean(axis=0)
    return LocalView.centered_on(calibration, center, LOCAL_PATCH_SIZE, LOCAL_FOCAL_PX)


def derive_marking_points(polygon, calibration=None):
    """Reduce a labeled 4-point slot polygon to its entrance marking-point
    pair + depth, for bootstrapping keypoint training data straight from
    existing whole-polygon labels (no new manual corner annotation needed
    for a first pass).

    ponytail: picks the shorter pair of opposite edges as the entrance --
    tuned for this project's slots (wider entrance than depth is wrong;
    verify against real corner labels if slot proportions differ).

    calibration: optional CalibrationModel. Raw fisheye pixel distances are
    radius-dependent (the same real depth measures fewer pixels near the
    frame edge, where this project's slots mostly sit) -- when given,
    entrance-edge/depth are measured in a small locally-rectified tangent
    plane around the polygon instead of raw pixels, which strips that
    radius-dependent foreshortening out. p1/p2 are still returned in raw
    pixel coordinates (matching every other label format in this project);
    depth is in local-rectified pixel units at LOCAL_FOCAL_PX, only
    meaningful when reconstruct_slot_quad() is given the same calibration.
    Default None keeps the original flat raw-pixel measurement.

    Returns (p1, p2, depth). p1/p2 are always ordered so that
    reconstruct_slot_quad(p1, p2, depth, calibration) reproduces this
    polygon -- the fixed clockwise-rotation convention in
    reconstruct_slot_quad only points the right way if every derived pair
    is normalized to it here.
    """
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.shape != (4, 2):
        raise ValueError(f"expected a 4-point polygon, got shape {poly.shape}")

    # coarse pass (centered on the whole polygon) just to pick which edge is
    # the entrance and which way it faces -- topology only, doesn't need to
    # match reconstruct_slot_quad's own view precisely.
    measure = poly if calibration is None else _local_view_at(calibration, poly).raw_to_local(poly)

    edge_lengths = [float(np.linalg.norm(measure[(i + 1) % 4] - measure[i])) for i in range(4)]
    entrance_idx = int(np.argmin(edge_lengths))
    i1, i2 = entrance_idx, (entrance_idx + 1) % 4
    far_a_idx, far_b_idx = (entrance_idx + 2) % 4, (entrance_idx + 3) % 4  # far_a opp. p2, far_b opp. p1

    if calibration is not None:
        # re-measure depth in a view centered on just the entrance edge --
        # the same one reconstruct_slot_quad(calibration=...) will use, so
        # the round trip lines up instead of drifting from two different
        # tangent planes.
        measure = _local_view_at(calibration, [poly[i1], poly[i2]]).raw_to_local(poly)

    depth = float((np.linalg.norm(measure[far_b_idx] - measure[i1]) +
                   np.linalg.norm(measure[far_a_idx] - measure[i2])) / 2)

    inward = _rotate90_cw(measure[i2] - measure[i1])
    facing_far_side = np.dot(
        inward, ((measure[far_a_idx] + measure[far_b_idx]) / 2) - ((measure[i1] + measure[i2]) / 2)) > 0
    if not facing_far_side:
        i1, i2 = i2, i1

    return poly[i1].tolist(), poly[i2].tolist(), depth


def reconstruct_slot_quad(p1, p2, depth, calibration=None):
    """Reconstruct a slot's full 4-point polygon from its 2 entrance
    marking points plus a known parking-depth prior.

    This is the geometric core of the marking-point redesign: replaces
    whole-polygon detection (fragile when a car partially occludes the
    slot) with 2-point detection + reconstruction, since entrance corners
    stay visible even when the rest of the slot doesn't. The p1->p2 order
    fixes direction: the slot is assumed to extend from the
    clockwise-rotated entrance edge, matching the convention
    derive_marking_points normalizes training labels to.

    calibration: optional CalibrationModel, same one derive_marking_points
    was given when measuring `depth`. A flat 90-degree rotation of raw
    fisheye pixels points the wrong way away from the image center (real
    straight lines curve in raw pixel space); when given, the rotation
    happens in a locally-rectified tangent plane around p1/p2 instead,
    where "perpendicular" means what it actually does in the real scene,
    and the result is mapped back through the lens model. Default None
    keeps the original flat-pixel-space rotation.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    if calibration is None:
        inward = _rotate90_cw(p2 - p1)
        norm = np.linalg.norm(inward)
        if norm < 1e-9:
            raise ValueError("p1 and p2 must be distinct points")
        inward = inward / norm * depth
        return np.array([p1, p2, p2 + inward, p1 + inward])

    view = _local_view_at(calibration, [p1, p2])
    p1_l, p2_l = view.raw_to_local(np.array([p1, p2]))
    inward = _rotate90_cw(p2_l - p1_l)
    norm = np.linalg.norm(inward)
    if norm < 1e-9:
        raise ValueError("p1 and p2 must be distinct points")
    inward = inward / norm * depth
    far2, far1 = view.local_to_raw(np.array([p2_l + inward, p1_l + inward]))
    return np.array([p1, p2, far2, far1])
