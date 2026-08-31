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


def train(data_yaml, base_model="yolov8n-seg.pt", epochs=100, device=None, hsv_v=0.6, mosaic=0.0,
          degrees=180.0, flipud=0.5):
    """hsv_v/mosaic default away from ultralytics' own defaults (0.4/1.0) for
    this project specifically:
    - hsv_v=0.6 (up from 0.4): stronger brightness/exposure jitter, aimed at
      the lighting/reflection variance HANDOFF.md documents (fluorescent
      CCTV lighting, glare off car glass/paint) -- this is the same problem
      class the pasted "RandomBrightnessContrast" suggestion targets, done
      via ultralytics' own native hyp instead of adding a new augmentation
      library dependency for one knob.
    - mosaic=0.0 (down from 1.0): ultralytics' default mosaic augmentation
      tiles 4 *different* images into one training sample. Each of our
      images is a single circular fisheye frame with one fixed optical
      center; tiling 4 of them together creates a training image with 4
      disconnected fake optical centers, unlike anything the model will
      ever see at real inference time. Left off unless a future dataset
      structure makes this assumption safe again.
    - degrees=180/flipud=0.5 (ultralytics defaults: 0/0): a ceiling-mounted
      top-down fisheye scene has no canonical "up" -- rotating the whole
      frame or flipping it vertically produces an equally valid scene, so
      these effectively multiply orientation coverage for free. Directly
      targets the observed weakness: edge slots at unusual orientations
      coming out flattened/diagonal.
    """
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, single_cls=True, device=device or _auto_device(),
                hsv_v=hsv_v, mosaic=mosaic, degrees=degrees, flipud=flipud)
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
    parser.add_argument("--hsv-v", type=float, default=0.6,
                         help="brightness/exposure jitter strength (0-1). Higher than ultralytics' own "
                              "default (0.4) for this project's lighting/reflection variance.")
    parser.add_argument("--mosaic", type=float, default=0.0,
                         help="mosaic augmentation probability (0-1). Off by default: it tiles 4 unrelated "
                              "fisheye frames together, which don't share one real optical center.")
    parser.add_argument("--degrees", type=float, default=180.0,
                         help="rotation augmentation range. Top-down fisheye has no canonical 'up', so full "
                              "rotation is a valid scene -- multiplies orientation coverage.")
    parser.add_argument("--flipud", type=float, default=0.5,
                         help="vertical flip probability -- also valid for a top-down scene.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = train(args.data, args.base_model, args.epochs, args.device, args.hsv_v, args.mosaic,
                  args.degrees, args.flipud)
    _save_checkpoint(model, args.output)
    print(f"trained -> {args.output}")


if __name__ == "__main__":
    main()
