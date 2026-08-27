import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def train(data_yaml, base_model="yolov8n-seg.pt", epochs=100):
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, single_cls=True)
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = train(args.data, args.base_model, args.epochs)
    _save_checkpoint(model, args.output)
    print(f"trained -> {args.output}")


if __name__ == "__main__":
    main()
