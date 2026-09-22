"""Evaluate STeF maps for ATC (hourly) and ETH/UCY (100-frame batches).

STeF-map stores a histogram over eight direction bins. To compare it with the
SWGMM-based methods (which return densities over orientation), the bin
probability is converted to a density by dividing by the bin width (2*pi/8).
STeF-map models orientation only, so its NLL is computed over orientation.

Examples:
    python evaluate_stef.py --dataset ATC
    python evaluate_stef.py --dataset ATC --hours 9 10
    python evaluate_stef.py --dataset ETHUCY --scenes eth hotel
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import compute_NLL_utils as nll_utils
from utils import load_dataset_config


ATC_TEST_FILES = ["atc/1028.csv", "atc/1031.csv", "atc/1104.csv"]
ATC_HOURS = range(9, 21)
ETHUCY_SCENES = ("eth", "hotel", "students003", "zara01")
MAP_COLUMNS = ["x", "y", *[f"theta_{i}" for i in range(1, 9)]]

NUM_BINS = 8
BIN_WIDTH = 2 * np.pi / NUM_BINS   # width of one direction bin in radians
DENSITY_FLOOR = 1e-12              # same density floor as the SWGMM-based methods


# STeF likelihood: a normalized histogram over eight circular direction bins.
def normalize_histogram(histogram):
    total = histogram.sum()
    if total == 0:
        return histogram
    return histogram / total


def get_nll_from_stef(cell_in_stefmap, point):
    """NLL of the observed orientation under the STeF histogram, as a density.

    Bins are centred at 0, 45, 90, ... degrees, i.e. bin 0 covers
    [-22.5, 22.5) degrees. The bin probability P_b is spread uniformly over
    the bin width, giving a piecewise-constant density P_b / BIN_WIDTH that
    integrates to one over [0, 2*pi).
    """
    hist = normalize_histogram(np.asarray(cell_in_stefmap[2:], dtype=float))
    index = int(round((NUM_BINS / (2 * np.pi)) * point["motion_angle"])) % NUM_BINS

    density = hist[index] / BIN_WIDTH
    return -np.log(max(density, DENSITY_FLOOR))


def _read_stef_map(map_file, bounds=None):
    data = pd.read_csv(map_file, header=None)
    data.columns = MAP_COLUMNS
    if bounds is not None:
        data = data[
            data["x"].between(*bounds["x"]) & data["y"].between(*bounds["y"])
        ]
    return data.to_numpy()


def read_stef_full_map_data(STeF_file):
    """Read an ATC map using the configured spatial bounds."""
    return _read_stef_map(STeF_file, bounds=load_dataset_config("ATC"))


def read_stef_full_map_data_ethucy(STeF_file):
    return _read_stef_map(STeF_file)


def find_nearest_cell(point, STeF_data, r_s):
    if len(STeF_data) == 0:
        return None
    location = np.array([point["x"], point["y"]])
    distances_squared = np.sum((STeF_data[:, :2] - location) ** 2, axis=1)
    index = np.argmin(distances_squared)
    if np.sqrt(distances_squared[index]) > r_s:
        return None
    return STeF_data[index]


def compute_nll(test_data, STeF_data, threshold):
    nlls = []
    not_find = 0
    for _, point in tqdm(test_data.iterrows(), total=len(test_data)):
        cell = find_nearest_cell(point, STeF_data, threshold)
        if cell is None:
            not_find += 1
            nlls.append(-np.log(DENSITY_FLOOR))
        else:
            nlls.append(get_nll_from_stef(cell, point))
    average_nll = np.mean(nlls) if nlls else float("nan")
    std_nll = np.std(nlls) if nlls else float("nan")
    return nlls, not_find, average_nll, std_nll


# Shared evaluation and output helpers (same output schema as evaluate_cliff.py).
def _statistics(nlls, not_find):
    return {
        "num_samples": len(nlls),
        "average_nll": float(np.mean(nlls)) if len(nlls) else float("nan"),
        "std_nll": float(np.std(nlls)) if len(nlls) else float("nan"),
        "not_find": not_find,
    }


def _save_log(log_file, stats):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        f"average_nll: {stats['average_nll']:.3f}, "
        f"std_nll: {stats['std_nll']:.3f}, not_find: {stats['not_find']}\n"
    )


def _evaluate_samples(test_data, map_data, threshold, sample_file, log_file):
    nlls, not_find, _, _ = compute_nll(test_data, map_data, threshold)
    if nlls is None:
        raise ValueError(f"No NLL results returned for {sample_file}")
    nlls = np.asarray(nlls, dtype=float)
    if nlls.ndim != 1 or len(nlls) != len(test_data):
        raise ValueError(f"Expected one NLL per test row for {sample_file}")

    sample_file = Path(sample_file)
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    out = test_data.copy()
    out["nll"] = nlls
    out.to_csv(sample_file, index=False)
    _save_log(log_file, _statistics(nlls, not_find))
    print(f"Saved per-sample NLLs to {sample_file}")
    return nlls, not_find


def _save_summary(results, summary_file, log_file):
    if not results:
        raise ValueError("No evaluation partitions selected")
    nlls = np.concatenate([values for values, _ in results])
    stats = _statistics(nlls, sum(count for _, count in results))
    summary_file = Path(summary_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(summary_file, index=False)
    _save_log(log_file, stats)
    print(f"Saved overall NLL summary to {summary_file}")
    print(f"Average NLL: {stats['average_nll']:.3f} | Std: {stats['std_nll']:.3f}")
    return stats


# ATC: evaluate each hourly map and pool all selected hours.
def evaluate_hour_stef(hour):
    cfg = load_dataset_config("ATC")
    x_min, x_max = cfg["x"]
    y_min, y_max = cfg["y"]
    map_data = _read_stef_map(f"MoDs/stef_atc/stef-map-{hour}.csv", bounds=cfg)
    test_data = nll_utils.read_test_data_with_hour(
        hour, datafile=ATC_TEST_FILES, dataset="ATC",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
    )
    return _evaluate_samples(
        test_data, map_data, 1.0,
        f"nll_results/stef_atc/stef-map-{hour}.csv",
        f"logs/stef_atc/stef-map-{hour}.txt",
    )


def evaluate_all_hours_stef(hours=ATC_HOURS):
    results = [evaluate_hour_stef(hour) for hour in hours]
    return _save_summary(
        results, "nll_results/stef_atc/summary.csv", "logs/stef_atc/summary.txt",
    )


# ETH/UCY: evaluate available maps in numeric batch order, with 100 frames per batch.
def evaluate_stef_ethucy(batch_num, version):
    map_data = read_stef_full_map_data_ethucy(
        f"MoDs/stef_ethucy/{version}/stef-map-{batch_num}.csv",
    )
    test_data = nll_utils.read_test_data_with_frame_stef(
        batch_num, datafile=f"eth_ucy/test/{version}.csv",
    )
    return _evaluate_samples(
        test_data, map_data, 1.0,
        f"nll_results/stef_ethucy/{version}/b{batch_num}.csv",
        f"logs/stef_ethucy/{version}/b{batch_num}.txt",
    )


def evaluate_all_batches_stef_ethucy(version="eth"):
    map_dir = Path("MoDs/stef_ethucy") / version
    batch_nums = sorted(
        int(path.stem[len("stef-map-"):])
        for path in map_dir.glob("stef-map-*.csv")
        if path.stem[len("stef-map-"):].isdigit()
    )
    if not batch_nums:
        raise FileNotFoundError(f"No batch maps found in {map_dir}")
    results = [evaluate_stef_ethucy(batch, version) for batch in batch_nums]
    return _save_summary(
        results, f"nll_results/stef_ethucy/{version}/summary.csv",
        f"logs/stef_ethucy/{version}.txt",
    )


# Command-line entry point; importing this module does not run evaluations.
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["ATC", "ETHUCY"], default="ETHUCY")
    parser.add_argument(
        "--scenes", nargs="+", choices=[*ETHUCY_SCENES, "students001"],
        help="ETH/UCY scenes (default: eth hotel students003 zara01)",
    )
    parser.add_argument(
        "--hours", nargs="+", type=int, choices=range(24),
        help="ATC hours (default: 9 through 20)",
    )
    args = parser.parse_args()
    if args.dataset == "ATC":
        if args.scenes is not None:
            parser.error("--scenes applies only to ETHUCY")
        evaluate_all_hours_stef(args.hours if args.hours is not None else ATC_HOURS)
    else:
        if args.hours is not None:
            parser.error("--hours applies only to ATC")
        for scene in args.scenes if args.scenes is not None else ETHUCY_SCENES:
            evaluate_all_batches_stef_ethucy(scene)


if __name__ == "__main__":
    main()


###### Run ######
# python3 evaluate_stef.py --dataset ATC
# python3 evaluate_stef.py --dataset ETHUCY