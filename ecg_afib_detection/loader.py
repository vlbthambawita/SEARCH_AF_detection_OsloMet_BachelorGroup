# loader.py
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset, Subset


class ECGDataset(Dataset):
    def __init__(self, data_dir):
        data = torch.load(Path(data_dir) / "data.pt", map_location="cpu")
        self.X = data["X"]
        self.y = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_kfold(data_dir, fs, val_fold, n_folds=5):
    dataset = ECGDataset(data_dir)
    df = pd.read_csv(Path(data_dir) / f"samples_{fs}hz.csv")

    train_idx = df[df.fold != val_fold].index.tolist()
    val_idx   = df[df.fold == val_fold].index.tolist()

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    num_classes = int(dataset.y.max().item() + 1)
    return train_ds, val_ds, num_classes
