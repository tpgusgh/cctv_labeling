import numpy as np
import cv2


class CalibrationModel:
    """Single-parameter equidistant fisheye <-> rectilinear mapping.

    ponytail: one scalar focal parameter fit by least-squares against
    user-clicked collinear points on a real straight line — upgrade to
    full cv2.fisheye.calibrate() + checkerboard captures if that data
    ever becomes available.
    """

    def __init__(self, cx: float, cy: float, f: float, radius=None):
        self.cx = float(cx)
        self.cy = float(cy)
        self.f = float(f)
        self.radius = radius

    def undistort_points(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        r_d = np.sqrt(dx ** 2 + dy ** 2)
        theta = r_d / self.f
        r_u = self.f * np.tan(theta)
        safe_r_d = np.where(r_d > 1e-9, r_d, 1.0)
        scale = np.where(r_d > 1e-9, r_u / safe_r_d, 1.0)
        return np.stack([self.cx + dx * scale, self.cy + dy * scale], axis=1)

    def distort_points(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        r_u = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan(r_u / self.f)
        r_d = self.f * theta
        safe_r_u = np.where(r_u > 1e-9, r_u, 1.0)
        scale = np.where(r_u > 1e-9, r_d / safe_r_u, 1.0)
        return np.stack([self.cx + dx * scale, self.cy + dy * scale], axis=1)

    def undistort_image(self, raw_bgr):
        h, w = raw_bgr.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        rectified_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        raw_sample_coords = self.distort_points(rectified_coords)
        map_x = raw_sample_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = raw_sample_coords[:, 1].reshape(h, w).astype(np.float32)
        return cv2.remap(raw_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)

    def redistort_image(self, rectified_bgr, output_shape):
        h, w = output_shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        raw_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        rectified_sample_coords = self.undistort_points(raw_coords)
        map_x = rectified_sample_coords[:, 0].reshape(h, w).astype(np.float32)
        map_y = rectified_sample_coords[:, 1].reshape(h, w).astype(np.float32)
        return cv2.remap(rectified_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)

    def pixel_to_ray(self, points):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = pts[:, 0] - self.cx
        dy = pts[:, 1] - self.cy
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = r / self.f
        phi = np.arctan2(dy, dx)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack([x, y, z], axis=1)

    def ray_to_pixel(self, rays):
        rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        rays = rays / norms
        theta = np.arccos(np.clip(rays[:, 2], -1.0, 1.0))
        phi = np.arctan2(rays[:, 1], rays[:, 0])
        r = self.f * theta
        dx = r * np.cos(phi)
        dy = r * np.sin(phi)
        return np.stack([self.cx + dx, self.cy + dy], axis=1)

    def to_dict(self):
        return {"model": "equidistant_1param", "center": [self.cx, self.cy], "f": self.f, "radius": self.radius}

    @classmethod
    def from_dict(cls, d):
        cx, cy = d["center"]
        return cls(cx, cy, d["f"], radius=d.get("radius"))


def fit(clicked_points, cx, cy, f_min=50.0, f_max=1000.0, n_coarse=50):
    pts = np.asarray(clicked_points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError(f"fit() needs at least 3 clicked collinear points, got {len(pts)}")

    def residual(f):
        model = CalibrationModel(cx, cy, f)
        undistorted = model.undistort_points(pts)
        centered = undistorted - undistorted.mean(axis=0)
        singular_values = np.linalg.svd(centered, full_matrices=False)[1]
        return singular_values[-1]

    candidates = np.linspace(f_min, f_max, n_coarse)
    residuals = [residual(f) for f in candidates]
    best_idx = int(np.argmin(residuals))
    if best_idx in (0, n_coarse - 1):
        raise ValueError(
            f"fit() optimum landed at the search boundary (f≈{candidates[best_idx]:.1f}); "
            f"the clicked points may not be truly collinear, or f_min/f_max need widening"
        )
    lo = candidates[max(best_idx - 1, 0)]
    hi = candidates[min(best_idx + 1, n_coarse - 1)]

    for _ in range(40):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if residual(m1) < residual(m2):
            hi = m2
        else:
            lo = m1

    return CalibrationModel(cx, cy, (lo + hi) / 2)


def _rotation_aligning_to_z(v):
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = np.dot(v, z)

    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
        return 2 * np.outer(axis, axis) - np.eye(3)

    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    axis = axis / s
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + K * s + (K @ K) * (1 - c)


class LocalView:
    def __init__(self, calibration, rotation, local_f, patch_size):
        self.calibration = calibration
        self.rotation = rotation
        self.local_f = local_f
        self.patch_size = patch_size

    @classmethod
    def centered_on(cls, calibration, center_raw_point, patch_size, local_f):
        ray = calibration.pixel_to_ray([center_raw_point])[0]
        rotation = _rotation_aligning_to_z(ray)
        return cls(calibration, rotation, local_f, patch_size)

    def raw_to_local(self, raw_points):
        rays = self.calibration.pixel_to_ray(raw_points)
        local_rays = rays @ self.rotation.T
        w, h = self.patch_size
        lx = self.local_f * local_rays[:, 0] / local_rays[:, 2] + w / 2.0
        ly = self.local_f * local_rays[:, 1] / local_rays[:, 2] + h / 2.0
        return np.stack([lx, ly], axis=1)

    def local_to_raw(self, local_points):
        pts = np.asarray(local_points, dtype=np.float64).reshape(-1, 2)
        w, h = self.patch_size
        x = (pts[:, 0] - w / 2.0) / self.local_f
        y = (pts[:, 1] - h / 2.0) / self.local_f
        z = np.ones_like(x)
        local_rays = np.stack([x, y, z], axis=1)
        local_rays = local_rays / np.linalg.norm(local_rays, axis=1, keepdims=True)
        world_rays = local_rays @ self.rotation
        return self.calibration.ray_to_pixel(world_rays)
