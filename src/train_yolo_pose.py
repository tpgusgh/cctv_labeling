import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def _auto_device():
    # ponytail: same MPS-not-auto-selected gap as train_yolo_seg.py --
    # ultralytics' own device selector defaults to CPU unless a device is
    # passed explicitly, even when torch.backends.mps.is_available().
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(data_yaml, base_model="yolov8n-pose.pt", epochs=100, device=None, hsv_v=0.6, mosaic=0.0):
    """Same non-default hyp choices as train_yolo_seg.py, for the same
    reasons (this project's lighting variance, and single-fisheye-frame
    images that mosaic's 4-image tiling doesn't make sense for)."""
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, single_cls=True, device=device or _auto_device(),
                hsv_v=hsv_v, mosaic=mosaic)
    return model


def _save_checkpoint(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(output_path))
    except AttributeError:
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        shutil.copy2(best, output_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a COCO-pretrained YOLOv8-pose checkpoint on an exported marking-point dataset.")
    parser.add_argument("--data", required=True, help="path to dataset.yaml from export_marking_points.py")
    parser.add_argument("--base-model", default="yolov8n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="models/yolov8_pose_marking_points.pt")
    parser.add_argument("--device", default=None,
                         help="torch device (0 for first CUDA GPU, 'mps', 'cpu'). "
                              "Default: auto-detect best available.")
    parser.add_argument("--hsv-v", type=float, default=0.6)
    parser.add_argument("--mosaic", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = train(args.data, args.base_model, args.epochs, args.device, args.hsv_v, args.mosaic)
    _save_checkpoint(model, args.output)
    print(f"trained -> {args.output}")


if __name__ == "__main__":
    main()
