import json
from pathlib import Path

from calibration import CalibrationModel


def raw_clicks_to_slot_polygon(raw_points, calibration):
    rectified = calibration.undistort_points(raw_points)
    return rectified.tolist()


class SlotConfig:
    def __init__(self, camera_id, image_width, image_height, calibration, slots, label_spec, rectified_size=None):
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.calibration = calibration
        self.slots = slots
        self.label_spec = label_spec
        if rectified_size is None:
            rectified_size = (image_width, image_height)
        self.rectified_size = tuple(rectified_size)

    def save(self, path):
        payload = {
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "calibration": self.calibration.to_dict(),
            "slots": self.slots,
            "label_spec": self.label_spec,
            "rectified_size": list(self.rectified_size),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        calibration = CalibrationModel.from_dict(payload["calibration"])
        rectified_size = payload.get("rectified_size")
        if rectified_size is None:
            rectified_size = (payload["image_width"], payload["image_height"])
        return cls(
            payload["camera_id"],
            payload["image_width"],
            payload["image_height"],
            calibration,
            payload["slots"],
            payload["label_spec"],
            rectified_size=rectified_size,
        )
