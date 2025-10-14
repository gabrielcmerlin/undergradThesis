from aeon.datasets import load_regression
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pywt
from skimage.transform import resize
from sktime.transformations.panel.rocket import Rocket
import os
import joblib

class DatasetManager:
    """
    Handles data loading in a dataset-agnostic way, allowing the use of 
    different datasets without modifying the main code.
    """

    def __init__(self, name, device, batch_size, is_transforming = True):
        self.dtype = torch.get_default_dtype()
        self.dataset_name = name
        self.batch_size = batch_size
        self.device = device
        self.is_trasforming = is_transforming
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

        if self.is_trasforming:
            self.X_train, self.y_train = self.transform(X_train, y_train)
            self.X_test, self.y_test = self.transform(X_test, y_test)
        else:
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

    def load_dataloader_for_training(self):
        """Create training and testing DataLoaders ready for use."""
        
        # Add channel dimension if required (e.g., for FCN).
        X_tr = self.X_train.unsqueeze(1) if self.X_train.ndim == 2 else self.X_train
        X_te = self.X_test.unsqueeze(1) if self.X_test.ndim == 2 else self.X_test

        train_ds = TensorDataset(X_tr, self.y_train)
        test_ds = TensorDataset(X_te, self.y_test)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=False)

        return train_loader, test_loader

class DatasetManagerTSERMamba(Dataset):
    """
    Dataset manager for TSERMamba model.
    Loads data using `aeon.datasets.load_regression` and computes:
        - CWT-transformed data
        - ROCKET features
    Returns (batch_x_cwt, batch_x_rocket, y)
    """

    def __init__(self, name, device, batch_size, rescale_size=64, wt_name='morl', projected_space=64):
        self.dataset_name = name
        self.device = device
        self.batch_size = batch_size
        self.rescale_size = rescale_size
        self.wt_name = wt_name
        self.projected_space = projected_space

        self.dtype = torch.get_default_dtype()

        self.load_data()
        self.prepare_cwt()
        self.prepare_rocket()

    def load_data(self):
        """Load train/test split from aeon and normalize"""
        X_train, y_train = load_regression(self.dataset_name, split="train")
        X_test, y_test = load_regression(self.dataset_name, split="test")

        self.dims = X_train.shape[1]

        # Replace NaNs
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        # Normalize per series
        X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / (X_train.std(axis=2, keepdims=True) + 1e-8)
        X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / (X_test.std(axis=2, keepdims=True) + 1e-8)

        self.seq_len = max(X_train.shape[1],X_test.shape[1])

        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.y_train = torch.tensor(y_train, dtype=self.dtype).unsqueeze(1)
        self.y_test = torch.tensor(y_test, dtype=self.dtype).unsqueeze(1)

    def prepare_cwt(self):
        """Compute CWT for all samples"""
        N, D = self.X_train_raw.shape[0], self.X_train_raw.shape[1]
        self.X_train_cwt = np.zeros((N, D, self.rescale_size, self.rescale_size), dtype=np.float32)
        for i in range(N):
            for d in range(D):
                coeffs, freqs = pywt.cwt(self.X_train_raw[i, d], np.arange(1, self.rescale_size+1), self.wt_name)
                self.X_train_cwt[i, d] = resize(coeffs, (self.rescale_size, self.rescale_size), mode='constant')

        N, D = self.X_test_raw.shape[0], self.X_test_raw.shape[1]
        self.X_test_cwt = np.zeros((N, D, self.rescale_size, self.rescale_size), dtype=np.float32)
        for i in range(N):
            for d in range(D):
                coeffs, freqs = pywt.cwt(self.X_test_raw[i, d], np.arange(1, self.rescale_size+1), self.wt_name)
                self.X_test_cwt[i, d] = resize(coeffs, (self.rescale_size, self.rescale_size), mode='constant')

        # Normalize CWT to [0,1]
        cwt_min, cwt_max = np.min(self.X_train_cwt), np.max(self.X_train_cwt)
        self.X_train_cwt = (self.X_train_cwt - cwt_min) / (cwt_max - cwt_min)
        self.X_test_cwt = (self.X_test_cwt - cwt_min) / (cwt_max - cwt_min)

    def prepare_rocket(self):
        """Compute ROCKET features per channel (like channel_token_mixing=0)"""
        rocket_dir = f"./rocket_models/{self.dataset_name}"
        os.makedirs(rocket_dir, exist_ok=True)

        N_train, D, _ = self.X_train_raw.shape
        N_test = self.X_test_raw.shape[0]

        # Cria espaço para armazenar features do ROCKET por canal
        self.X_train_rocket = np.zeros((N_train, D, self.projected_space), dtype=np.float32)
        self.X_test_rocket = np.zeros((N_test, D, self.projected_space), dtype=np.float32)

        for d in range(D):
            rocket_model_path = os.path.join(rocket_dir, f"rocket_channel_{d}.pkl")

            # Carrega modelo se existir, senão treina
            if os.path.exists(rocket_model_path):
                rocket = joblib.load(rocket_model_path)
            else:
                rocket = Rocket(num_kernels=self.projected_space // 2, normalise=True)
                rocket.fit(self.X_train_raw[:, d:d+1])  # shape (N_train, 1)
                joblib.dump(rocket, rocket_model_path)

            # Transforma canal individualmente
            self.X_train_rocket[:, d, :] = rocket.transform(self.X_train_raw[:, d:d+1])
            self.X_test_rocket[:, d, :] = rocket.transform(self.X_test_raw[:, d:d+1])

        # Normaliza globalmente
        r_min, r_max = np.min(self.X_train_rocket), np.max(self.X_train_rocket)
        self.X_train_rocket = (self.X_train_rocket - r_min) / (r_max - r_min)
        self.X_test_rocket = (self.X_test_rocket - r_min) / (r_max - r_min)

    def load_dataloader_for_training(self):
        """Return train/test dataloaders yielding (X_cwt, X_rocket, y)"""

        # Convert CWT and ROCKET features to torch tensors
        X_train_cwt = torch.from_numpy(self.X_train_cwt).float() if isinstance(self.X_train_cwt, np.ndarray) else self.X_train_cwt.detach().clone().float()
        X_test_cwt = torch.from_numpy(self.X_test_cwt).float() if isinstance(self.X_test_cwt, np.ndarray) else self.X_test_cwt.detach().clone().float()
       
        # Convert ROCKET features to NumPy arrays if needed
        X_train_rocket = self.X_train_rocket.values if hasattr(self.X_train_rocket, "values") else self.X_train_rocket
        X_test_rocket = self.X_test_rocket.values if hasattr(self.X_test_rocket, "values") else self.X_test_rocket

        X_train_rocket = torch.from_numpy(X_train_rocket).float() if isinstance(X_train_rocket, np.ndarray) else X_train_rocket.detach().clone().float()
        X_test_rocket = torch.from_numpy(X_test_rocket).float() if isinstance(X_test_rocket, np.ndarray) else X_test_rocket.detach().clone().float()

        # --- RAW ---
        X_train_raw = torch.from_numpy(self.X_train_raw).float().unsqueeze(1) if self.X_train_raw.ndim == 2 else torch.from_numpy(self.X_train_raw).float()
        X_test_raw = torch.from_numpy(self.X_test_raw).float().unsqueeze(1) if self.X_test_raw.ndim == 2 else torch.from_numpy(self.X_test_raw).float()

        # Ensure y tensors are float
        y_train = self.y_train.detach().clone().float()
        y_test = self.y_test.detach().clone().float()

        # Create TensorDatasets
        train_ds = TensorDataset(X_train_cwt, X_train_rocket, X_train_raw, y_train)
        test_ds = TensorDataset(X_test_cwt, X_test_rocket, X_test_raw, y_test)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=False, collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=False, collate_fn=lambda x: collate_fn(x, max_len=self.seq_len))

        return train_loader, test_loader

def collate_fn(data, max_len=None):
    """
    Builds mini-batches for TSER Mamba.
    Input: list of tuples (x_cwt, x_rocket, x_raw, label)
    Output: XCWT, Rocket_and_RAW, targets
    """

    batch_size = len(data)
    x_cwt, x_rocket, x_raw, labels = zip(*data)

    # --- CWT ---
    XCWT = torch.stack(x_cwt, dim=0)  # (B, D, H, W)

    # --- ROCKET ---
    XROCKET = torch.stack(x_rocket, dim=0)  # (B, D, projected_space)

    # --- RAW features ---
    lengths = [X.shape[-1] for X in x_raw]
    if max_len is None:
        max_len = max(lengths)

    D = x_raw[0].shape[0]
    X = torch.zeros(batch_size, D, max_len)  # (B, D, L)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :, :end] = x_raw[i][:, :end]

    # --- Concatenar Rocket + RAW ao longo da dimensão das features ---
    Rocket_and_RAW = torch.cat([XROCKET, X], dim=2)  # (B, D, projected_space + L)

    # --- Labels ---
    targets = torch.stack(labels, dim=0)

    return XCWT, Rocket_and_RAW, targets