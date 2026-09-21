"""Compare hourly ATC NLLs.

Run: python3 p_value_atc.py
     python3 p_value_atc.py --results-dir nll_results --output results/atc.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from p_value import KEYS, paired_ci, read_results


HOURS = range(9, 21)
METHODS = {
    "ours": "Ours",
    "online": "Online CLiFF-map",
    "cliff": "CLiFF-map",
    "stef": "STeF-map",
}


def result_path(root, hour, method):
    if method == "ours":
        return root / "nemo_atc_hour" / f"atc-{hour}.csv"
    if method == "online":
        return root / "online_cliff_atc" / f"ATC1024_{hour}_{hour+1}_online_online.csv"
    if method == "cliff":
        return root / "cliff_map_atc" / f"cliff-map-{hour}.csv"
    if method == "stef":
        return root / "stef_atc" / f"stef-map-{hour}.csv"
    raise ValueError(f"Unknown method: {method}")


def build_table(root):
    all_rows = []
    for hour in HOURS:
        frames = {method: read_results([result_path(root, hour, method)], method)
                  for method in METHODS}
        matched = frames["ours"]
        for method in METHODS:
            if method != "ours":
                matched = matched.merge(frames[method], on=KEYS, how="inner", validate="one_to_one")
        before_cleaning = len(matched)
        matched = matched.loc[np.isfinite(matched[list(METHODS)]).all(axis=1)]
        counts = ", ".join(f"{method}={len(frame)}" for method, frame in frames.items())
        print(f"ATC {hour}-{hour+1}: input rows ({counts}); "
              f"matched={before_cleaning}; finite={len(matched)}")
        if any(len(frame) != len(matched) for frame in frames.values()):
            print("  NOTE: table uses only the common finite samples across all four methods.")
        # Retain only NLL columns to limit memory when pooling all hours.
        all_rows.append(matched[list(METHODS)].copy())

    pooled = pd.concat(all_rows, ignore_index=True)
    if len(pooled) < 2:
        raise ValueError(f"ATC: only {len(pooled)} valid common samples")
    print(f"Total valid common samples: {len(pooled)}")

    rows = []
    for method, label in METHODS.items():
        values = pooled[method]
        row = {
            "Method": label,
            "NLL↓": f"{values.mean():.3f} ± {values.std(ddof=1):.3f}",
            "NLL reduction (vs Ours)": "-",
            "95% CI": "-",
        }
        if method != "ours":
            reduction, (low, high) = paired_ci(values, pooled["ours"])
            test = stats.ttest_rel(values, pooled["ours"], alternative="greater")
            row["NLL reduction (vs Ours)"] = f"{reduction:+.3f}"
            row["95% CI"] = f"[{low:.3f}, {high:.3f}]"
            print(f"  {label}: reduction={reduction:+.6f}, "
                  f"95% CI=[{low:.6f}, {high:.6f}], "
                  f"t={test.statistic:.6f}, p={test.pvalue:.3e}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "nll_results")
    parser.add_argument("--output", type=Path, help="Output CSV (default: RESULTS_DIR/atc_comparison.csv)")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.results_dir / "atc_comparison.csv"
    inputs = {result_path(args.results_dir, hour, method).resolve()
              for hour in HOURS for method in METHODS}
    if output.resolve() in inputs:
        raise ValueError("Output must not overwrite an input NLL file")

    table = build_table(args.results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nSaved table to {output}")


if __name__ == "__main__":
    main()