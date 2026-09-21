"""Compare ETH/UCY NLLs and write a publication-style table to one CSV.

ATC comparisons and historical results are preserved in p_value_atc.py.

Run: python p_value.py
     python p_value.py --results-dir nll_results --output results/ethucy.csv

Columns are ETH, HOTEL, UNIV (students003), and ZARA (zara01).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


KEYS = ["time", "x", "y", "speed", "motion_angle"]
SCENES = {"eth": "ETH", "hotel": "HOTEL", "students003": "UNIV", "zara01": "ZARA"}
METHODS = {
    "cliff": "CLiFF-map",
    "online": "Online CLiFF-map",
    "stef": "STeF-map",
}
BATCH_COUNTS = {
    "online": {"eth": 3, "hotel": 3, "students003": 2, "zara01": 2},
    "stef": {"eth": 15, "hotel": 12, "students003": 6, "zara01": 9},
}


def paired_ci(a, b, alpha=0.05):
    """Mean paired reduction and its two-sided Student-t confidence interval."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Expected equal-length 1D paired arrays with at least two samples")
    d = a - b
    if not np.isfinite(d).all():
        raise ValueError("Paired differences must be finite")
    mean_diff = d.mean()
    margin = stats.t.ppf(1 - alpha / 2, df=len(d) - 1) * stats.sem(d, ddof=1)
    return mean_diff, (mean_diff - margin, mean_diff + margin)


def result_paths(root, scene, method):
    if method == "ours":
        return [root / f"nemo_{scene}" / f"{scene}-all.csv"]
    if method == "cliff":
        return [root / "cliff_map_ethucy" / f"{scene}.csv"]
    folder = "online_cliff_ethucy" if method == "online" else "stef_ethucy"
    return [root / folder / scene / f"b{i}.csv" for i in range(BATCH_COUNTS[method][scene])]


def read_results(paths, method):
    """Concatenate all expected batches, checking sample identity globally."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(KEYS + ["nll"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frames.append(frame[KEYS + ["nll"]])
    result = pd.concat(frames, ignore_index=True)
    if result[KEYS].isna().any().any():
        raise ValueError(f"{method}: missing sample keys in {paths}")
    if result.duplicated(KEYS).any():
        raise ValueError(f"{method}: duplicate sample keys, including across batches, in {paths}")
    return result.rename(columns={"nll": method})


def compare_scene(root, scene):
    method_ids = ["ours", *METHODS]
    frames = {method: read_results(result_paths(root, scene, method), method)
              for method in method_ids}
    matched = frames["ours"]
    for method in METHODS:
        matched = matched.merge(frames[method], on=KEYS, how="inner", validate="one_to_one")
    before_cleaning = len(matched)
    matched = matched.loc[np.isfinite(matched[method_ids]).all(axis=1)]
    if len(matched) < 2:
        raise ValueError(f"{scene}: only {len(matched)} valid common samples")

    counts = ", ".join(f"{method}={len(frame)}" for method, frame in frames.items())
    print(f"{SCENES[scene]}: input rows ({counts}); matched={before_cleaning}; finite={len(matched)}")
    if any(len(frame) != len(matched) for frame in frames.values()):
        print("  NOTE: table uses only the common finite samples across all four methods.")

    cells = [f"{matched.ours.mean():.3f} ± {matched.ours.std(ddof=1):.3f}"]
    for method, label in METHODS.items():
        a, b = matched[method], matched["ours"]
        mean_diff, (low, high) = paired_ci(a, b)
        test = stats.ttest_rel(a, b, alternative="greater")
        cells.extend([
            f"{a.mean():.3f} ± {a.std(ddof=1):.3f}",
            f"{mean_diff:+.3f}",
            f"[{low:.3f}, {high:.3f}]",
        ])
        print(f"  {label}: reduction={mean_diff:+.6f}, 95% CI=[{low:.6f}, {high:.6f}], "
              f"t={test.statistic:.6f}, p={test.pvalue:.3e}")
    return cells


def build_table(root):
    rows = ["Ours"]
    for label in METHODS.values():
        rows.extend([label, "Reduction vs Ours", "95% CI"])
    table = pd.DataFrame({"Method": rows})
    for scene, column in SCENES.items():
        table[column] = compare_scene(root, scene)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "nll_results")
    parser.add_argument("--output", type=Path, help="Output CSV (default: RESULTS_DIR/ethucy_comparison.csv)")
    args = parser.parse_args()
    table = build_table(args.results_dir)
    output = args.output if args.output is not None else args.results_dir / "ethucy_comparison.csv"

    inputs = {p.resolve() for scene in SCENES for method in ["ours", *METHODS]
              for p in result_paths(args.results_dir, scene, method)}
    if output.resolve() in inputs:
        raise ValueError("Output must not overwrite an input NLL file")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nSaved table to {output}")


if __name__ == "__main__":
    main()
