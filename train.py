import os
import time
import numpy as np
import random
import argparse
import torch

from dataset import HumanMotionTimeDataset, ETHUCYDataset, get_dataloader_onlytrainval
import models
import loss_funcs
from utils import normalize_coords_space_time, normalize_coords_siren, load_dataset_config, get_exp_name, DATASET_CHOICES
from torch.utils.tensorboard import SummaryWriter


def set_random_seed(seed):
    seed = seed if seed >= 0 else random.randint(0, 2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_args():
    parser = argparse.ArgumentParser(description="Train motion dynamics model")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default="ATC",
        help="Dataset to use: ATC, or an ETH/UCY scene as <ETH|UCY>-<version> (default: ATC)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["time_grid", "fourier", "siren"],
        default="siren",
        help="Model type to use: time_grid, fourier, or siren"
    )

    parser.add_argument("--num-components", type=int, default=None,
                        help="Override the number of GMM components in the dataset config")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Override the square grid size in the dataset config")
    args = parser.parse_args()
    if args.num_components is not None and args.num_components < 1:
        parser.error("--num-components must be at least 1")
    if args.grid_size is not None and args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    return args


if __name__ == "__main__":
    args = get_args()
    
    set_random_seed(42)
    
    model_name = args.model

    # Per-dataset settings (normalization bounds + train: batch_size / grid_size / siren_variant)
    dataset_cfg = load_dataset_config(args.dataset)
    train_cfg = dataset_cfg.get("train", {})
    if args.num_components is not None:
        train_cfg["num_components"] = args.num_components
    if args.grid_size is not None:
        train_cfg["grid_size"] = [args.grid_size, args.grid_size]
    batch_size = train_cfg.get("batch_size", 256)

    model = models.build_model(model_name, train_cfg)

    exp_name = get_exp_name(model_name, args.dataset)
    if args.num_components is not None or args.grid_size is not None:
        grid = model.grid_size
        grid_label = str(grid[0]) if grid[0] == grid[1] else f"{grid[0]}x{grid[1]}"
        exp_name = f"{exp_name.replace('_', '-')}-com{model.num_components}-grid{grid_label}"
    print(f"Dataset: {args.dataset} | Model: {model_name} -> {type(model).__name__} grid {tuple(model.grid_size)} "
          f"| batch {batch_size} | outputs -> models/{exp_name}, runs/{exp_name}")

    log_dir = f"runs/{exp_name}/{int(time.time())}"
    os.makedirs(f"models/{exp_name}", exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    best_valid_loss = float('inf')

    device = torch.device('cuda')

    ##### For dataset #####
    if args.dataset == "ATC":
        dataset_file_path = "atc/1024.csv"
        dataset = HumanMotionTimeDataset(dataset_file_path)
    elif args.dataset.startswith(("ETH-", "UCY-")):
        # "ETH-eth" -> "eth", "UCY-students003" -> "students003", etc.
        version = args.dataset.split("-", 1)[1]
        dataset_file_path = f"eth_ucy/train/{version}.csv"
        dataset = ETHUCYDataset(dataset_file_path)
    ###########################

    train_loader, val_loader = get_dataloader_onlytrainval(dataset, batch_size=batch_size)

    model = model.to(device)
    criterion = loss_funcs.NLLGMMLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            if model_name in ["time_grid", "fourier"]:
                norm_inputs = normalize_coords_space_time(inputs, dataset_cfg)
            elif model_name == "siren":
                norm_inputs = normalize_coords_siren(inputs, dataset_cfg)

            output, _ = model(norm_inputs)
            loss = criterion(output, targets)
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.flush()
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                if model_name in ["time_grid", "fourier"]:
                    norm_inputs = normalize_coords_space_time(inputs, dataset_cfg)
                elif model_name == "siren":
                    norm_inputs = normalize_coords_siren(inputs, dataset_cfg)
                
                output, coords = model(norm_inputs)
                loss = criterion(output, targets)
                valid_loss += loss.item()
                
        valid_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss}")
        writer.add_scalar("Loss/Validation", valid_loss, epoch) 
        writer.flush()
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"models/{exp_name}/best.pt")

        # save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"models/{exp_name}/{epoch}.pt")

    writer.close()
    ##########################################################################################
