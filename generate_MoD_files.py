############# for generating MoD files from trained models #############
import datetime, pytz
import torch
import argparse
import numpy as np
import pandas as pd
import models
import time
import os

from utils import normalize_coords_space_time, normalize_coords_siren

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

if __name__ == "__main__":
    args = get_args()

    model_name = args.model

    if model_name == "time_grid":
        exp_name = f"distri_gmm_feature_time_v2"
        model = models.MoDGMMFeatureTimeModel(input_size=3, num_components=3)
    elif model_name == "fourier":
        exp_name = f"distri_gmm_feature_ff_time_v2"
        model = models.MoDGMMFeatureFFModel(input_size=3, num_components=3)
    elif model_name == "siren":
        exp_name = f"distri_gmm_siren_v2"
        model = models.MoDGMMSirenHybridModel(input_size=3, num_components=3)

    model_dir = f"models/{exp_name}/best.pt"

    # Load the saved weights
    model.load_state_dict(torch.load(model_dir, weights_only=True))
    model.eval()

    ## get ATC map grids
    atc_file = "atc/1024.csv"
    atc_data = pd.read_csv(atc_file, header=None, names=["time", "person_id", "x", "y", "speed", "motion_angle"])

    x_min, x_max, y_min, y_max, step = -60, 80, -40, 20, 1
    x_centers = np.arange(x_min, x_max, step)
    y_centers = np.arange(y_min, y_max, step)

    ################# get locations from x_centers and y_centers #############
    # locations = [[x, y] for x in x_centers for y in y_centers]
    # example_locations = torch.tensor(locations, dtype=torch.float32)
    ###########################################################################

    ####################  OR: get map grids from another MoD #######################
    cliff_mod_file = "MoDs/cliff/cliff.csv"
    MoD_columns = ["x", "y", "motion_angle", "velocity",
                        "cov4", "cov2", "cov3", "cov1", "weight",
                        "observation_ratio", "motion_ratio"]
    cliff_data = pd.read_csv(cliff_mod_file, header=None, names=MoD_columns)
    cliff_data["x"] = cliff_data["x"].round(3)
    cliff_data["y"] = cliff_data["y"].round(3)
    xy_unique = cliff_data[['x', 'y']].drop_duplicates()
    locations = xy_unique.values.tolist()
    example_locations = torch.tensor(locations, dtype=torch.float32)
    ###########################################################################
    
    for hour in range(9, 21):

        tz = pytz.timezone("Asia/Tokyo")
        dt_tokyo = tz.localize(datetime.datetime(2012, 10, 24, hour, 30, 0))
        epoch_time = dt_tokyo.timestamp()
        print(epoch_time)

        t_seconds = epoch_time
        t_col = torch.full((example_locations.shape[0], 1),
                        fill_value=t_seconds,
                        dtype=torch.float32,
                        device=example_locations.device)

        example_inputs = torch.cat([example_locations, t_col], dim=1)  # (N,3)
        
        if model_name in ["time_grid", "fourier"]:
            norm_inputs = normalize_coords_space_time(example_inputs)
        elif model_name == "siren":
            norm_inputs = normalize_coords_siren(example_inputs)

        print(f"Number of locations: {example_locations.size(0)}")

        start_time = time.time()

        with torch.no_grad():
            GMM_params, _ = model(norm_inputs)

        end_time = time.time()
        inference_time = end_time - start_time

        print(f"Inference time: {inference_time:.4f} seconds")

        ########## Save to csv files ##########
        data = []

        num_components = GMM_params.shape[1]

        for j in range(example_locations.size(0)):
            # density = densities[j]
            for i in range(num_components):
                weight = GMM_params[j, i, 0]
                speed_mean = GMM_params[j, i, 1]
                angle_mean = GMM_params[j, i, 2]
                speed_var = GMM_params[j, i, 3]
                angle_var = GMM_params[j, i, 4]
                corr_coef = GMM_params[j, i, 5]
                
                row = [float(example_locations[j, 0]), float(example_locations[j, 1]), float(speed_mean), float(angle_mean), float(speed_var), float(angle_var), float(corr_coef), float(weight)]
                data.append(row)

        df = pd.DataFrame(data, columns=["x", "y", "mean_speed", "mean_motion_angle", "var_speed", "var_motion_angle", "coef", "weight"])

        # output_csv_file = f"MoDs/{exp_name}.csv"
        output_csv_file = f"MoDs/{exp_name}/{hour}.csv"
        os.makedirs(f"MoDs/{exp_name}", exist_ok=True)
        df.to_csv(output_csv_file, index=False)
        print(f"Saved predictions to {output_csv_file}")
        ########################################