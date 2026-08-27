import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def _auto_device():
    # ponytail: ultralytics does NOT auto-select MPS (Apple Silicon) even
    # when available -- its own device selector defaults to CPU unless a
    # device is passed explicitly (verified: select_device("") -> "cpu" on
    # this machine despite torch.backends.mps.is_available() == True).
    # CUDA IS auto-selected by ultralytics already, but picking it here too
    # keeps this function's result meaningful on either platform.
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(data_yaml, base_model="yolov8n-seg.pt", epochs=100, device=None):
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, single_cls=True, device=device or _auto_device())
    return model


def _save_checkpoint(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(output_path))
    except AttributeError:
        # ponytail: fallback for ultralytics versions where YOLO has no
        # top-level .save() -- the trainer always writes its best checkpoint
        # to <save_dir>/weights/best.pt regardless.
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        shutil.copy2(best, output_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a COCO-pretrained YOLOv8-seg checkpoint on an exported parking-slot dataset.")
    parser.add_argument("--data", required=True, help="path to dataset.yaml from export_yolo_dataset.py")
    parser.add_argument("--base-model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="models/yolov8_seg_slots.pt")
    parser.add_argument("--device", default=None,
                         help="torch device (0 for first CUDA GPU, 'mps', 'cpu'). "
                              "Default: auto-detect best available.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = train(args.data, args.base_model, args.epochs, args.device)
    _save_checkpoint(model, args.output)
    print(f"trained -> {args.output}")


if __name__ == "__main__":
    main()
