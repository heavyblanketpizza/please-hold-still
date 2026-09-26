"""Compare probe results of two or more evaluated models (before vs after training).

    uv run python scripts/compare_models.py baseline run1

Prints a table and saves a chart to <data root>/eval/compare__baseline__run1.png.
Each tag must have been evaluated first with scripts/evaluate_encoder.py.
Every model after the first is compared with the first, answer by answer on
the same test scans (a paired comparison), using each tag's predictions.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from please_hold_still import paths
from please_hold_still.probe import (
    comparison_table,
    mismatched_volumes_message,
    paired_differences,
)
from please_hold_still.viz import plot_probe_comparison


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tags", nargs="+", help="evaluated models, first = reference (e.g. baseline)")
    p.add_argument("--data-root", type=Path, default=None, help="default: $PLEASE_HOLD_STILL_DATA")
    p.add_argument("--out", type=Path, default=None, help="chart path (.png)")
    args = p.parse_args()

    root = paths.ensure_data_root(args.data_root)
    summaries = {}
    for tag in args.tags:
        path = paths.eval_dir(root) / tag / "results.json"
        if not path.is_file():
            print(f"{path} not found. Run: scripts/evaluate_encoder.py --tag {tag} ...")
            return 1
        summaries[tag] = json.loads(path.read_text())
    mismatch = mismatched_volumes_message(summaries)
    if mismatch:
        print(mismatch)
        print("Evaluate every model on the same data first (scripts/evaluate_encoder.py).")
        return 1
    results = {tag: s["probes"] for tag, s in summaries.items()}

    # Paired comparison against the first model, where both have saved predictions.
    first, *others = args.tags
    pred_paths = {tag: paths.eval_dir(root) / tag / "predictions.csv" for tag in args.tags}
    missing = [tag for tag, path in pred_paths.items() if not path.is_file()]
    paired = {}
    if first not in missing:
        reference = pd.read_csv(pred_paths[first])
        for tag in others:
            if tag not in missing:
                paired[tag] = paired_differences(reference, pd.read_csv(pred_paths[tag]))

    table = comparison_table(results, paired)
    if table.empty:
        print("No probe has results for every model (too few labelled test volumes?).")
        print("See the 'skipped' reasons printed by evaluate_encoder.py.")
        return 1
    print(table.to_string(index=False))
    if paired:
        print(
            f"\nchange: difference from {first}; [range] = paired 95% range (both models "
            "answer the same test scans).\n"
            '"within noise": that range includes 0, so the difference could be luck.'
        )
    if len(paired) < len(others):
        print(
            '\nwithout [range]: "within noise" means the two models\' 95% ranges overlap '
            "(stricter)."
        )
        print(
            f"No predictions.csv for {', '.join(missing)}. Re-run evaluate_encoder.py "
            "--tag <tag> to save them; it reuses the saved features (about a minute)."
        )
    out = args.out or paths.eval_dir(root) / f"compare__{'__'.join(args.tags)}.png"
    print(f"chart: {plot_probe_comparison(results, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
