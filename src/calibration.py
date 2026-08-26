import numpy as np
import cv2


class CalibrationModel:
    """Single-parameter equidistant fisheye <-> rectilinear mapping.

    ponytail: one scalar focal parameter fit by least-squares against
    user-clicked collinear points on a real straight line — upgrade to
    full cv2.fisheye.calibrate() + checkerboard captures if that data
    ever becomes available.
    """

    def __init__(self, cx: float, cy: float, f: float):
        self.cx = float(cx)
        self.cy = float(cy)
        self.f = float(f)

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

    def to_dict(self):
        return {"model": "equidistant_1param", "center": [self.cx, self.cy], "f": self.f}

    @classmethod
    def from_dict(cls, d):
        cx, cy = d["center"]
        return cls(cx, cy, d["f"])
