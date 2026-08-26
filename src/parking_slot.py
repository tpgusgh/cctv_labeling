import json
from pathlib import Path

from calibration import CalibrationModel


class SlotConfig:
    def __init__(self, camera_id, image_width, image_height, calibration, slots, label_spec):
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.calibration = calibration
        self.slots = slots
        self.label_spec = label_spec

    def save(self, path):
        payload = {
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "calibration": self.calibration.to_dict(),
            "slots": self.slots,
            "label_spec": self.label_spec,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        calibration = CalibrationModel.from_dict(payload["calibration"])
        return cls(
            payload["camera_id"],
            payload["image_width"],
            payload["image_height"],
            calibration,
            payload["slots"],
            payload["label_spec"],
        )
