import argparse
import sys

import review_store
import slot_classifier

DEFAULT_LABELS_PATH = "review/labels.jsonl"
DEFAULT_MODEL_PATH = "models/slot_classifier.joblib"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Retrain the slot classifier from human accept/reject review feedback.")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="path to review/labels.jsonl")
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH, help="path to write the trained model")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    labels = review_store.load_labels(args.labels)
    if not labels:
        print(f"no labels found at {args.labels} -- review some candidates first, nothing trained")
        return

    try:
        model = slot_classifier.train(labels)
    except ValueError as e:
        print(f"cannot train: {e}")
        sys.exit(1)

    slot_classifier.save(model, args.output)
    print(f"trained on {len(labels)} label(s) -> {args.output}")


if __name__ == "__main__":
    main()
