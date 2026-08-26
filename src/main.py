import argparse

from pipeline import run


def build_parser():
    parser = argparse.ArgumentParser(
        description="Composite a label onto one raw CCTV frame (sub-project 1 test harness).")
    parser.add_argument("--config", required=True, help="path to camera config JSON")
    parser.add_argument("--image", required=True, help="path to raw input frame")
    parser.add_argument("--slot-id", required=True, help="slot id from the config to place the label in")
    parser.add_argument("--candidate-u", type=float, required=True, help="candidate label center, normalized 0-1")
    parser.add_argument("--candidate-v", type=float, required=True, help="candidate label center, normalized 0-1")
    parser.add_argument("--output", required=True, help="path to write the final PNG")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args.config, args.image, args.slot_id, (args.candidate_u, args.candidate_v), args.output)


if __name__ == "__main__":
    main()
