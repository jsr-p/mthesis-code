import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# Define a PyTorch Dataset
class BinaryDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.x = torch.randn(num_samples, num_features)
        self.y = torch.round(torch.rand(num_samples))  # random binary labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def normalize(X: torch.Tensor):
    """Taken from pyg."""
    X = X - X.min()
    return X.div_(X.sum(dim=-1, keepdim=True).clamp_(min=1.0))


class BinaryClassificationData(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = normalize(X)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create a PyTorch DataLoader
def create_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)


def tensors_from_numpy(X, y) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(X).float(), torch.from_numpy(y.ravel()).float()


def dataset_from_numpy(X, y) -> Dataset:
    return BinaryClassificationData(*tensors_from_numpy(X, y))


def binary_data(iris: bool = False) -> BinaryClassificationData:
    """Returns binary data"""
    if iris:
        from sklearn import datasets

        iris = datasets.load_iris()  # type: ignore
        X: np.ndarray = iris.data[:, :2]  # type: ignore
        y: np.ndarray = iris.target  # type: ignore
        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
    else:
        data = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        ).values
        X = data[:, :-1]
        y = data[:, -1]
    return dataset_from_numpy(X, y)


def split_dataset(dataset, batch_size=128):
    train, val, test = random_split(dataset, [0.6, 0.2, 0.2])
    train_loader = create_data_loader(train, batch_size=batch_size)
    val_loader = create_data_loader(val, batch_size=batch_size)
    test_loader = create_data_loader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader
