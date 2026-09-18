import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
    

class HumanMotionTimeDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        if isinstance(file_path, list):
            dfs = []
            for f in file_path:
                df = pd.read_csv(f, header=None, names=["time", "person_id", "x", "y", "speed", "motion_angle"])
                df['motion_angle'] = np.mod(df['motion_angle'], 2 * np.pi)
                dfs.append(df)
            self.data = pd.concat(dfs, ignore_index=True)
        else:
            self.data = pd.read_csv(file_path, header=None, names=["time", "person_id", "x", "y", "speed", "motion_angle"])
            self.data['motion_angle'] = np.mod(self.data['motion_angle'], 2 * np.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        motion_angle = row["motion_angle"] % (2 * np.pi)
        
        input_data = torch.tensor([row["x"], row["y"], row["time"]], dtype=torch.float32)
        target_data = torch.tensor([row["speed"], motion_angle], dtype=torch.float32)

        sample = {"input": input_data, "target": target_data}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ETHUCYDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform
        self.data = pd.read_csv(file_path)
        # rename columns, from frame to time
        self.data = self.data.rename(columns={"frame": "time", "pedestrian_ID": "person_id", "pos_x": "x", "pos_y": "y"})
        self.data['speed'] = np.linalg.norm(self.data[['v_x', 'v_y']], axis=1)
        self.data['motion_angle'] = np.arctan2(self.data['v_y'], self.data['v_x'])
        self.data['motion_angle'] = np.mod(self.data['motion_angle'], 2 * np.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        motion_angle = row["motion_angle"] % (2 * np.pi)
        
        input_data = torch.tensor([row["x"], row["y"], row["time"]], dtype=torch.float32)
        target_data = torch.tensor([row["speed"], motion_angle], dtype=torch.float32)

        sample = {"input": input_data, "target": target_data}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader_onlytrainval(dataset, batch_size):
    total_indices = np.arange(len(dataset))

    # Split into train (90%) and val (10%) directly
    train_indices, val_indices = train_test_split(
        total_indices, test_size=0.1
    )

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_size = len(train_loader.dataset)
    val_size   = len(val_loader.dataset)
    print("Train size:", train_size, " Val size:", val_size)

    return train_loader, val_loader