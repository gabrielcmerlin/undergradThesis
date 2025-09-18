from aeon.datasets import load_regression
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class DatasetManager:
    """
    Handles data loading in a dataset-agnostic way, allowing the use of 
    different datasets without modifying the main code.
    """

    def __init__(self, name, device, batch_size=32):
        self.dtype = torch.get_default_dtype()
        self.dataset_name = name
        self.batch_size = batch_size
        self.device = device
        self.load_data()

    def transform(self, X, y):
        """Convert numpy arrays to PyTorch tensors with correct shape for regression."""

        X = torch.as_tensor(X, dtype=self.dtype)
        y = torch.as_tensor(np.array(y, dtype=float), dtype=self.dtype).unsqueeze(1)  # shape [N,1]
        return X, y

    def load_data(self):
        """Load training and testing data and convert them to tensors."""

        X_train, y_train = load_regression(name=self.dataset_name, split='train') # type: ignore
        X_test, y_test = load_regression(name=self.dataset_name, split='test') # type: ignore

        # Replace NaN values with 0.
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Normalize each time series.
        X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / (X_train.std(axis=2, keepdims=True) + 1e-8)
        X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / (X_test.std(axis=2, keepdims=True) + 1e-8)

        # Remove redundant singleton dimension if present.
        if X_train.ndim == 3 and X_train.shape[1] == 1:
            X_train = X_train[:, 0, :]
        if X_test.ndim == 3 and X_test.shape[1] == 1:
            X_test = X_test[:, 0, :]

        self.X_train, self.y_train = self.transform(X_train, y_train)
        self.X_test, self.y_test = self.transform(X_test, y_test)

    def load_dataloader_for_training(self):
        """Create training and testing DataLoaders ready for use."""
        
        # Add channel dimension if required (e.g., for FCN).
        X_tr = self.X_train.unsqueeze(1) if self.X_train.ndim == 2 else self.X_train
        X_te = self.X_test.unsqueeze(1) if self.X_test.ndim == 2 else self.X_test

        train_ds = TensorDataset(X_tr, self.y_train)
        test_ds = TensorDataset(X_te, self.y_test)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=16)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return train_loader, test_loader