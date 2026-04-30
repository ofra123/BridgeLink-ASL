from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.io_utils import read_json, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Accepted for consistency; not required.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir, Path.cwd())
    baseline = read_json(out_dir / "baseline_metrics.json")
    finetuned = read_json(out_dir / "finetuned_metrics.json")
    keys = sorted(set(baseline) | set(finetuned))
    rows = []
    for key in keys:
        before = baseline.get(key)
        after = finetuned.get(key)
        delta = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else ""
        rows.append({"metric": key, "baseline": before, "finetuned": after, "delta": delta})

    widths = {
        "metric": max(len("metric"), *(len(str(r["metric"])) for r in rows)),
        "baseline": max(len("baseline"), *(len(str(r["baseline"])) for r in rows)),
        "finetuned": max(len("finetuned"), *(len(str(r["finetuned"])) for r in rows)),
        "delta": max(len("delta"), *(len(str(r["delta"])) for r in rows)),
    }
    header = f"{'metric':<{widths['metric']}}  {'baseline':>{widths['baseline']}}  {'finetuned':>{widths['finetuned']}}  {'delta':>{widths['delta']}}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{str(r['metric']):<{widths['metric']}}  "
            f"{str(r['baseline']):>{widths['baseline']}}  "
            f"{str(r['finetuned']):>{widths['finetuned']}}  "
            f"{str(r['delta']):>{widths['delta']}}"
        )

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "baseline", "finetuned", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison to {csv_path}")


if __name__ == "__main__":
    main()
