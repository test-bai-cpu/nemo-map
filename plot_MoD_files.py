# ########### Plotting the results from the csv file ###########
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from plot_utils import plot_cliff_map_with_weight

def get_args():
    parser = argparse.ArgumentParser(description="Train motion dynamics model")

    parser.add_argument(
        "--model",
        type=str,
        choices=["time_grid", "fourier", "siren"],
        default="siren",
        help="Model type to use: time_grid, fourier, or siren"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        choices=["max", "all"],
        default="max",
        help="Plot version: max or all"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    model_name = args.model

    version = args.version

    if model_name == "time_grid":
        exp_name = f"distri_gmm_feature_time_v2"
    elif model_name == "fourier":
        exp_name = f"distri_gmm_feature_ff_time_v2"
    elif model_name == "siren":
        exp_name = f"distri_gmm_siren_v2"

    for hour in range(9,21):
        print(hour)
        output_csv_file = f"MoDs/{exp_name}/{hour}.csv"
        df = pd.read_csv(output_csv_file)
        df['mean_motion_angle'] = np.mod(df['mean_motion_angle'], 2 * np.pi)
        
        df = df[(df['x'] >= -60) & (df['x'] <= 80) & (df['y'] >= -40) & (df['y'] <= 20)]
        
        cliff_map_data = df.to_numpy()
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10, 6), dpi=100)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.subplot(111, facecolor='white')
        img = plt.imread("atc-map/localization_grid_white.jpg")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])
        plt.axis('off')
        plot_cliff_map_with_weight(cliff_map_data, mod="ours", version=version)
        os.makedirs(f"MoDs/{exp_name}/{version}_png", exist_ok=True)
        plt.savefig(f"MoDs/{exp_name}/{version}_png/{hour}.png", bbox_inches='tight', pad_inches=0)
        # ##################################################