"""Evaluate CLiFF and online CLiFF maps for ATC or ETH/UCY.

Examples:
    python evaluate_cliff.py --dataset ATC --map-type cliff
    python evaluate_cliff.py --dataset ATC --map-type online
    python evaluate_cliff.py --dataset ETHUCY --map-type cliff --scenes eth hotel
    python evaluate_cliff.py --dataset ETHUCY --map-type online
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import compute_NLL_utils as nll_utils
from utils import load_dataset_config


ATC_TEST_FILES = ["atc/1028.csv", "atc/1031.csv", "atc/1104.csv"]
ATC_HOURS = range(9, 21)
ETHUCY_SCENES = ("eth", "hotel", "students003", "zara01")


# Shared evaluation and output helpers.
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
    nlls, not_find, _, _ = nll_utils.compute_nll(test_data, map_data, threshold)
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


# ATC: both map types are evaluated by hour.
def _evaluate_atc_hour(hour, online=False):
    cfg = load_dataset_config("ATC")
    x_min, x_max = cfg["x"]
    y_min, y_max = cfg["y"]
    if online:
        folder = "online_cliff_atc"
        stem = f"ATC1024_{hour}_{hour + 1}_online_online"
        reader = nll_utils.read_cliff_map_data_obs_version_online
    else:
        folder = "cliff_map_atc"
        stem = f"cliff-map-{hour}"
        reader = nll_utils.read_cliff_map_data_obs_version

    map_data = reader(f"MoDs/{folder}/{stem}.csv")
    map_data = map_data[
        map_data["x"].between(x_min, x_max)
        & map_data["y"].between(y_min, y_max)
    ]
    test_data = nll_utils.read_test_data_with_hour(
        hour, datafile=ATC_TEST_FILES, dataset="ATC",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
    )
    return _evaluate_samples(
        test_data, map_data, 1.0,
        f"nll_results/{folder}/{stem}.csv", f"logs/{folder}/{stem}.txt",
    )


def evaluate_hour_cliff(hour):
    """Evaluate one ATC CLiFF hour; retain the existing NLL-array return value."""
    return _evaluate_atc_hour(hour)[0]


def evaluate_hour_online_cliff(hour):
    """Evaluate one ATC online CLiFF hour."""
    return _evaluate_atc_hour(hour, online=True)[0]


def evaluate_all_hours_cliff(online=False, hours=ATC_HOURS):
    """Evaluate and summarize ATC hours for the selected map type."""
    folder = "online_cliff_atc" if online else "cliff_map_atc"
    results = [_evaluate_atc_hour(hour, online=online) for hour in hours]
    return _save_summary(
        results, f"nll_results/{folder}/summary.csv", f"logs/{folder}/summary.txt",
    )


# ETH/UCY: CLiFF uses one map per scene; online CLiFF uses frame batches.
def evaluate_cliff_ethucy(file_name="eth"):
    map_data = nll_utils.read_cliff_map_data_python(f"MoDs/cliff_map_ethucy/{file_name}.csv")
    test_data = nll_utils.read_test_data(
        datafile=f"eth_ucy/test/{file_name}.csv", dataset="ETHUCY",
    )
    log_file = f"logs/ethucy_cliff_map/{file_name}.txt"
    result = _evaluate_samples(
        test_data, map_data, 1.0,
        f"nll_results/cliff_map_ethucy/{file_name}.csv", log_file,
    )
    return _save_summary(
        [result], f"nll_results/cliff_map_ethucy/{file_name}/summary.csv", log_file,
    )


def evaluate_online_cliff_ethucy(batch_num, file_name="eth"):
    map_data = nll_utils.read_cliff_map_data_obs_version_online(
        f"MoDs/online_cliff_ethucy/{file_name}/b{batch_num}_online.csv",
    )
    test_data = nll_utils.read_test_data_with_frame(
        batch_num, datafile=f"eth_ucy/test/{file_name}.csv", dataset="ETHUCY",
    )
    return _evaluate_samples(
        test_data, map_data, 10.0,
        f"nll_results/online_cliff_ethucy/{file_name}/b{batch_num}.csv",
        f"logs/ethucy_online_cliff/{file_name}/b{batch_num}.txt",
    )


def evaluate_all_batches_online_cliff_ethucy(file_name="eth"):
    map_dir = Path("MoDs/online_cliff_ethucy") / file_name
    batch_nums = sorted(
        int(path.stem[1:-len("_online")])
        for path in map_dir.glob("b*_online.csv")
        if path.stem[1:-len("_online")].isdigit()
    )
    if not batch_nums:
        raise FileNotFoundError(f"No batch maps found in {map_dir}")
    results = [evaluate_online_cliff_ethucy(batch, file_name) for batch in batch_nums]
    return _save_summary(
        results, f"nll_results/online_cliff_ethucy/{file_name}/summary.csv",
        f"logs/ethucy_online_cliff/{file_name}.txt",
    )


# Command-line entry point. Defaults preserve the previous online ETH/UCY run.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ATC", "ETHUCY"], default="ETHUCY")
    parser.add_argument("--map-type", choices=["cliff", "online"], default="online")
    parser.add_argument("--scenes", nargs="+", choices=[*ETHUCY_SCENES, "students001"],
                        help="ETH/UCY scenes (default: eth hotel students003 zara01)")
    parser.add_argument("--hours", nargs="+", type=int, choices=range(24),
                        help="ATC hours (default: 9 through 20)")
    args = parser.parse_args()
    if args.dataset == "ATC":
        if args.scenes is not None:
            parser.error("--scenes applies only to ETHUCY")
        evaluate_all_hours_cliff(online=args.map_type == "online",
                                 hours=args.hours if args.hours is not None else ATC_HOURS)
    else:
        if args.hours is not None:
            parser.error("--hours applies only to ATC")
        evaluate_scene = (evaluate_all_batches_online_cliff_ethucy
                          if args.map_type == "online" else evaluate_cliff_ethucy)
        for scene in args.scenes if args.scenes is not None else ETHUCY_SCENES:
            evaluate_scene(scene)


if __name__ == "__main__":
    main()
