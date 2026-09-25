"""Compare probe results of two or more evaluated models (before vs after training).

    uv run python scripts/compare_models.py baseline run1

Prints a table and saves a chart to <data root>/eval/compare__baseline__run1.png.
Each tag must have been evaluated first with scripts/evaluate_encoder.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from please_hold_still import paths
from please_hold_still.probe import comparison_table
from please_hold_still.viz import plot_probe_comparison


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tags", nargs="+", help="evaluated models, first = reference (e.g. baseline)")
    p.add_argument("--data-root", type=Path, default=None, help="default: $PLEASE_HOLD_STILL_DATA")
    p.add_argument("--out", type=Path, default=None, help="chart path (.png)")
    args = p.parse_args()

    root = paths.ensure_data_root(args.data_root)
    results = {}
    for tag in args.tags:
        path = paths.eval_dir(root) / tag / "results.json"
        if not path.is_file():
            print(f"{path} not found. Run: scripts/evaluate_encoder.py --tag {tag} ...")
            return 1
        results[tag] = json.loads(path.read_text())["probes"]

    table = comparison_table(results)
    if table.empty:
        print("No probe has results for every model (too few labelled test volumes?).")
        print("See the 'skipped' reasons printed by evaluate_encoder.py.")
        return 1
    print(table.to_string(index=False))
    print('\n"within noise": the 95% ranges overlap, so the difference could be luck.')
    out = args.out or paths.eval_dir(root) / f"compare__{'__'.join(args.tags)}.png"
    print(f"chart: {plot_probe_comparison(results, out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
