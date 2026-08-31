"""One-command retrain: export -> fine-tune -> evaluate -> promote-if-better.

Run this whenever enough new review feedback has accumulated (rule of thumb:
100+ new accept/reject decisions since the last run):

    .venv/bin/python src/retrain_yolo.py

It fine-tunes from the current production checkpoint, then compares the
candidate against production on config recall (primary) and junk rate
(secondary). The production model at models/yolov8_seg_slots_production.pt
is replaced ONLY if the candidate wins; the loser is kept in models/archive/
either way, so nothing is ever lost. The web app picks the new model up on
its next restart (./run_web.sh).

Also retrains the classical-CV slot classifier (cheap, always safe).
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import review_store

PROJECT_ROOT = review_store.PROJECT_ROOT
PRODUCTION = PROJECT_ROOT / "models" / "yolov8_seg_slots_production.pt"
ARCHIVE_DIR = PROJECT_ROOT / "models" / "archive"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base-model", default=None,
                         help="checkpoint to fine-tune from (default: the current production model)")
    parser.add_argument("--force-promote", action="store_true",
                         help="replace production even if the candidate scores worse (not recommended)")
    args = parser.parse_args(argv)

    if not PRODUCTION.exists():
        sys.exit(f"production checkpoint missing: {PRODUCTION}\n"
                 f"copy your current best model there first, e.g.\n"
                 f"  cp models/yolov8_seg_slots_v6.pt {PRODUCTION}")

    base = args.base_model or str(PRODUCTION)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = ARCHIVE_DIR / f"candidate-{stamp}.pt"

    with tempfile.TemporaryDirectory(prefix="yolo_dataset_") as dataset_dir:
        print(f"[1/4] exporting dataset -> {dataset_dir}")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "export_yolo_dataset.py"),
                        "--output", dataset_dir], check=True)

        print(f"[2/4] fine-tuning from {base} ({args.epochs} epochs) -- takes a while")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "train_yolo_seg.py"),
                        "--data", f"{dataset_dir}/dataset.yaml",
                        "--base-model", base,
                        "--epochs", str(args.epochs),
                        "--degrees", "0", "--flipud", "0",  # fixed cameras: orientation is fixed too
                        "--output", str(candidate)], check=True)

    print("[3/4] evaluating candidate vs production")
    import evaluate_seg_model
    prod_score = evaluate_seg_model.evaluate(str(PRODUCTION))
    cand_score = evaluate_seg_model.evaluate(str(candidate))
    for name, r in (("production", prod_score), ("candidate", cand_score)):
        print(f"  {name}: recall {r['config_recall_hits']}/{r['config_slots']} ({r['recall']:.1%}), "
              f"junk {r['junk_rate']:.1%}")

    better = (cand_score["recall"], -cand_score["junk_rate"]) > (prod_score["recall"], -prod_score["junk_rate"])
    if better or args.force_promote:
        backup = ARCHIVE_DIR / f"production-before-{stamp}.pt"
        shutil.copy2(PRODUCTION, backup)
        shutil.copy2(candidate, PRODUCTION)
        print(f"[4/4] PROMOTED: candidate -> {PRODUCTION.name} (old model kept at {backup.name})")
        print("      restart the web app (./run_web.sh) to pick it up")
    else:
        print(f"[4/4] NOT promoted: candidate did not beat production (kept at {candidate.name})")

    print("retraining slot classifier (always safe)")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "src" / "train_from_reviews.py")], check=True)


if __name__ == "__main__":
    main()
