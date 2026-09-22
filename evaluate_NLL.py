import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from loss_funcs import wrapped_GMM_nll
from utils import normalize_coords_space_time, normalize_coords_siren, load_dataset_config, get_exp_name, get_scene_name, DATASET_CHOICES
import compute_NLL_utils
import models
import os
import time


device     = "cuda"   # or "cpu"
batch_size = 8192



def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained motion dynamics model (NLL on the test split)")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default="ATC",
        help="Dataset to evaluate on: ATC, or an ETH/UCY scene as <ETH|UCY>-<version> (default: ATC)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["time_grid", "fourier", "siren"],
        default="siren",
        help="Model type to use: time_grid, fourier, or siren"
    )

    args = parser.parse_args()
    return args


@torch.inference_mode()
def evaluate(model_name, model, df: pd.DataFrame, norm_cfg):
    inputs = torch.tensor(df[['x','y','time']].values, dtype=torch.float32)
    targets = torch.tensor(df[['speed','motion_angle']].values, dtype=torch.float32)

    ds = TensorDataset(inputs, targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True)

    model.eval().to(device)

    all_nll = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if model_name in ["time_grid", "fourier"]:
            xb_norm = normalize_coords_space_time(xb, norm_cfg)          # (B,3) on device
        elif model_name == "siren":
            xb_norm = normalize_coords_siren(xb, norm_cfg)

        GMM_params, _ = model(xb_norm)          # (B,K,6)

        nll_b = wrapped_GMM_nll(GMM_params, yb, reduction='none')  # (B,)
        all_nll.append(nll_b.cpu())

    nll = torch.cat(all_nll).numpy()
    return float(nll.mean()), float(nll.std()), nll


def evaluate_hour(hour, model_name="siren", dataset_name="ATC"):

    ################ Config #################
    test_data_file = [
            "atc/1028.csv",
            "atc/1031.csv",
            "atc/1104.csv",
        ]

    if model_name == "time_grid":
        model = models.MoDGMMFeatureTimeModel(input_size=3, num_components=3)
    elif model_name == "fourier":
        model = models.MoDGMMFeatureFFModel(input_size=3, num_components=3)
    elif model_name == "siren":
        model = models.MoDGMMSirenHybridModel(input_size=3, num_components=3)

    exp_name = get_exp_name(model_name, dataset_name)
    model_file = f"models/{exp_name}/best.pt"
    save_per_sample_outdir = f"nll_results/{exp_name}_hour"
    os.makedirs(save_per_sample_outdir, exist_ok=True)
    ###########################################

    test_data = compute_NLL_utils.read_test_data_with_hour(hour, datafile=test_data_file, dataset=dataset_name)

    state = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    norm_cfg = load_dataset_config(dataset_name)
    mean_nll, std_nll, nll_vec = evaluate(model_name, model, test_data, norm_cfg)

    out = test_data.copy()
    out['nll'] = nll_vec
    out.to_csv(f"{save_per_sample_outdir}/atc-{hour}.csv", index=False)
    print(f"Saved per-sample NLLs to {save_per_sample_outdir}")
    
    # print(f"Average NLL: {mean_nll:.6f} | Std: {std_nll:.6f}")

    # file_name = f"logs/{exp_name}_atc_hour/atc-{hour}.txt"
    # os.makedirs(f"logs/{exp_name}_atc_hour", exist_ok=True)
    # with open(file_name, "w") as f:
    #     f.write(f"average_nll: {mean_nll}, std_nll: {std_nll}\n")


def evaluate_all(model_name, dataset_name, num_component=None, grid_size=None):

    ################ Config #################
    scene = get_scene_name(dataset_name)          # "atc", "eth", "hotel", "zara01", ...
    dataset_cfg = load_dataset_config(dataset_name)   # bounds + train settings (model variant / grid size)

    if dataset_name == "ATC":
        test_data_file = [
                "atc/1028.csv",
                "atc/1031.csv",
                "atc/1104.csv",
            ]
    else:
        test_data_file = f"eth_ucy/test/{scene}.csv"

    train_cfg = dataset_cfg.get("train", {})
    if num_component is not None:
        train_cfg["num_components"] = num_component
    if grid_size is not None:
        train_cfg["grid_size"] = [grid_size, grid_size]
    model = models.build_model(model_name, train_cfg)   # same architecture as train.py

    exp_name = get_exp_name(model_name, dataset_name)
    if num_component is not None or grid_size is not None:
        exp_name = f"{exp_name.replace('_', '-')}-com{model.num_components}-grid{grid_size}"
    model_file = f"models/{exp_name}/best.pt"
    save_per_sample_outdir = f"nll_results/{exp_name}"
    save_per_sample_file = f"{save_per_sample_outdir}/{scene}-all.csv"   # ATC keeps "atc-all.csv"
    os.makedirs(save_per_sample_outdir, exist_ok=True)
    print(f"Dataset: {dataset_name} | Model: {model_name} -> {type(model).__name__} grid {tuple(model.grid_size)} "
          f"| checkpoint: {model_file} | output: {save_per_sample_file}")
    ###########################################

    # Filter the test split to the same spatial extent the model was normalized with.
    norm_cfg = dataset_cfg
    (x_min, x_max), (y_min, y_max) = norm_cfg["x"], norm_cfg["y"]
    test_data = compute_NLL_utils.read_test_data(datafile=test_data_file, dataset=dataset_name,
                                                 x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    state = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    mean_nll, std_nll, nll_vec = evaluate(model_name, model, test_data, norm_cfg)

    out = test_data.copy()
    out['nll'] = nll_vec
    out.to_csv(save_per_sample_file, index=False)
    print(f"Saved per-sample NLLs to {save_per_sample_file}")
    
    print(f"Average NLL: {mean_nll:.3f} | Std: {std_nll:.3f}")

    # return {"num_components": model.num_components,
    #         "average_nll": mean_nll, "std_nll": std_nll}
    
    # return {"grid_size": grid_size,
    #         "average_nll": mean_nll, "std_nll": std_nll}


if __name__ == "__main__":
    args = get_args()
    model_name = args.model
    
    evaluate_all(model_name, args.dataset)

    # for hour in range(9,21):
    #     evaluate_hour(hour)

    #### for ablation study of num_components ####
    # grid_size = 64
    # results = []
    # for num_component in range(1, 16):
    #     results.append(evaluate_all(model_name, args.dataset, num_component, grid_size))
    # os.makedirs("results", exist_ok=True)
    # file_name = "results/atc_num_components.csv"
    # pd.DataFrame(results).to_csv(file_name, index=False)
    # print(f"Saved component comparison to {file_name}")

    #### for ablation study of grid size ####
    num_component = 3
    results = []
    for grid_size in [8, 16, 32, 64, 128, 256]:
        results.append(evaluate_all(model_name, args.dataset, num_component, grid_size))
    os.makedirs("results", exist_ok=True)
    file_name = "results/atc_grid_size.csv"
    pd.DataFrame(results).to_csv(file_name, index=False)
    print(f"Saved grid size comparison to {file_name}")