import argparse

from pipeline import run, run_auto, run_auto_all


def build_parser():
    parser = argparse.ArgumentParser(
        description="Composite a label onto a parking slot in a CCTV image.")
    parser.add_argument("--config", required=True, help="path to camera config JSON")
    parser.add_argument("--image", required=True, help="path to raw input frame")
    parser.add_argument("--slot-id", help="slot id from the config to place the label in (ignored with --auto-all)")
    parser.add_argument("--candidate-u", type=float, help="candidate label center, normalized 0-1 (ignored with --auto/--auto-all)")
    parser.add_argument("--candidate-v", type=float, help="candidate label center, normalized 0-1 (ignored with --auto/--auto-all)")
    parser.add_argument("--output", required=True, help="path to write the final PNG")
    parser.add_argument("--auto", action="store_true",
                         help="auto-detect the windshield and compute label position/size instead of using --candidate-u/-v (single slot, requires --slot-id)")
    parser.add_argument("--auto-all", action="store_true",
                         help="auto-detect and label every slot in the config that has a visible windshield, in one output image (no --slot-id needed)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.auto_all:
        results = run_auto_all(args.config, args.image, args.output)
        for slot_id, status in results.items():
            print(f"slot '{slot_id}': {status}")
        return

    if args.slot_id is None:
        raise SystemExit("--slot-id is required unless --auto-all is set")

    if args.auto:
        result = run_auto(args.config, args.image, args.slot_id, args.output)
        if result is None:
            print(f"slot '{args.slot_id}': no visible windshield, skipped (no output written)")
        return

    if args.candidate_u is None or args.candidate_v is None:
        raise SystemExit("--candidate-u and --candidate-v are required unless --auto/--auto-all is set")

    run(args.config, args.image, args.slot_id, (args.candidate_u, args.candidate_v), args.output)


if __name__ == "__main__":
    main()
