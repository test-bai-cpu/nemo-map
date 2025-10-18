import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from loss_funcs import wrapped_GMM_nll
from utils import normalize_coords_space_time, normalize_coords_siren
import compute_NLL_utils
import models
import os
import time


device     = "cuda"   # or "cpu"
batch_size = 8192



def get_args():
    parser = argparse.ArgumentParser(description="Train motion dynamics model")

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
def evaluate(model_name, model, df: pd.DataFrame):
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
            xb_norm = normalize_coords_space_time(xb)          # (B,3) on device
        elif model_name == "siren":
            xb_norm = normalize_coords_siren(xb)

        GMM_params, _ = model(xb_norm)          # (B,K,6)

        nll_b = wrapped_GMM_nll(GMM_params, yb, reduction='none')  # (B,)
        all_nll.append(nll_b.cpu())

    nll = torch.cat(all_nll).numpy()
    return float(nll.mean()), float(nll.std()), nll


def evaluate_hour(hour):

    ################ Config #################
    dataset_name = "ATC"

    test_data_file = [
            "atc/1028.csv",
            "atc/1031.csv",
            "atc/1104.csv",
        ]

    if model_name == "time_grid":
        exp_name = f"distri_gmm_feature_time"
        model = models.MoDGMMFeatureTimeModel(input_size=3, num_components=3)
    elif model_name == "fourier":
        exp_name = f"distri_gmm_feature_ff_time"
        model = models.MoDGMMFeatureFFModel(input_size=3, num_components=3)
    elif model_name == "siren":
        exp_name = f"distri_gmm_siren"
        model = models.MoDGMMSirenHybridModel(input_size=3, num_components=3)

    model_file = f"models/{exp_name}/best.pt"
    save_per_sample_outdir = f"nll_results/{exp_name}"
    os.makedirs(save_per_sample_outdir, exist_ok=True)
    ###########################################

    test_data = compute_NLL_utils.read_test_data_with_hour(hour, datafile=test_data_file, dataset=dataset_name)

    state = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    mean_nll, std_nll, nll_vec = evaluate(model_name, model, test_data)

    out = test_data.copy()
    out['nll'] = nll_vec
    out.to_csv(f"{save_per_sample_outdir}/atc-{hour}.csv", index=False)
    print(f"Saved per-sample NLLs to {save_per_sample_outdir}")
    
    # print(f"Average NLL: {mean_nll:.6f} | Std: {std_nll:.6f}")

    file_name = f"results/{exp_name}/atc-{hour}.txt"
    os.makedirs(f"results/{exp_name}", exist_ok=True)
    with open(file_name, "w") as f:
        f.write(f"average_nll: {mean_nll}, std_nll: {std_nll}\n")


# for hour in range(9,21):
#     evaluate_hour(hour)
    


def evaluate_all(model_name):

    ################ Config #################
    dataset_name = "ATC"

    test_data_file = [
            "atc/1028.csv",
            "atc/1031.csv",
            "atc/1104.csv",
        ]

    if model_name == "time_grid":
        exp_name = f"distri_gmm_feature_time"
        model = models.MoDGMMFeatureTimeModel(input_size=3, num_components=3)
    elif model_name == "fourier":
        exp_name = f"distri_gmm_feature_ff_time"
        model = models.MoDGMMFeatureFFModel(input_size=3, num_components=3)
    elif model_name == "siren":
        exp_name = f"distri_gmm_siren"
        model = models.MoDGMMSirenHybridModel(input_size=3, num_components=3)

    model_file = f"models/{exp_name}/best.pt"
    save_per_sample_outdir = f"nll_results/{exp_name}"
    os.makedirs(save_per_sample_outdir, exist_ok=True)
    ###########################################

    test_data = compute_NLL_utils.read_test_data(datafile=test_data_file, dataset=dataset_name)

    state = torch.load(model_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    mean_nll, std_nll, nll_vec = evaluate(model_name, model, test_data)

    out = test_data.copy()
    out['nll'] = nll_vec
    out.to_csv(f"{save_per_sample_outdir}/atc-all.csv", index=False)
    print(f"Saved per-sample NLLs to {save_per_sample_outdir}")
    
    print(f"Average NLL: {mean_nll:.6f} | Std: {std_nll:.6f}")

    # file_name = f"results/{exp_name}/atc-all.txt"
    # os.makedirs(f"results/{exp_name}", exist_ok=True)
    # with open(file_name, "w") as f:
    #     f.write(f"average_nll: {mean_nll}, std_nll: {std_nll}\n")


args = get_args()
model_name = args.model
evaluate_all(model_name)