import os
import time
import numpy as np
import random
import argparse
import torch

from dataset import HumanMotionTimeDataset, get_dataloader_onlytrainval
import models
import loss_funcs
from utils import normalize_coords_space_time, normalize_coords_siren
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
    
    set_random_seed(42)
    
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

    log_dir = f"runs/{exp_name}/{int(time.time())}"
    os.makedirs(f"models/{exp_name}", exist_ok=True)

    batch_size = 256
    
    writer = SummaryWriter(log_dir=log_dir)
    best_valid_loss = float('inf')

    device = torch.device('cuda')

    dataset_file_path = "atc/1024.csv"
    dataset = HumanMotionTimeDataset(dataset_file_path)

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
                norm_inputs = normalize_coords_space_time(inputs)
            elif model_name == "siren":
                norm_inputs = normalize_coords_siren(inputs)

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
                    norm_inputs = normalize_coords_space_time(inputs)
                elif model_name == "siren":
                    norm_inputs = normalize_coords_siren(inputs)
                
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