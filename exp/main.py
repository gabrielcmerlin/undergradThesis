import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch

import json
import gc

from aeon.regression.convolution_based import RocketRegressor
from train_utils import calculate_metrics, scatter_ytrue_ypred

# Add BASE folder (parent of exp/) to sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Parser import Parser
from DatasetManager import DatasetManager, DatasetManagerTSERMamba
from models.ModelManager import ModelManager
from training_config import get_training_params
from train_utils import (
    train_model,
    evaluate_model,
    scatter_ytrue_ypred,
    train_TSERMamba,
    evaluate_TSERMamba,
)

EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = '../outputs/'

# Fix random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def wrap_results(metrics, model_name, dataset_name, elapsed, best_loss=None):
    results_data_dir = {
            'model': [], 'dataset': [], 'mse': [],
            'mae': [], 'r2': [], 'rmse': [],
            'best_train_loss': [], 'time': []
    }
    
    # Scatter plot.
    os.makedirs(f'{RESULTS_DIR}scatter/{model_name}', exist_ok=True)
    scatter_ytrue_ypred(metrics['y_true'], metrics['y_pred'],
                title=f"{dataset_name}",
                save_path=f'{RESULTS_DIR}scatter/{model_name}/{model_name}_{dataset_name}.png')

    # Store results.
    results_data_dir['model'].append(model_name)
    results_data_dir['dataset'].append(dataset_name)
    results_data_dir['mse'].append(metrics['mse'])
    results_data_dir['mae'].append(metrics['mae'])
    results_data_dir['r2'].append(metrics['r2'])
    results_data_dir['rmse'].append(metrics['rmse'])
    results_data_dir['best_train_loss'].append(best_loss)
    results_data_dir['time'].append(elapsed)

    os.makedirs(f"{RESULTS_DIR}results", exist_ok=True)
    pd.DataFrame(results_data_dir).to_csv(
        f"{RESULTS_DIR}results/{model_name}_{EXPERIMENT_DATE}.csv",
        index=False
    )

def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract datasets and models.
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])
    BATCH_SIZE = config.get("batch_size", 16)
    BATCH_DIVISIONS = config.get("batch_divisions", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    for model_name in MODELS:
        print(f"\n=== Model: {model_name} ===")

        # Decide which functions and dataset class to use based on the model.
        if model_name.lower() == "tsermamba":
            train_fn = train_TSERMamba
            eval_fn = evaluate_TSERMamba
            dataset_class = DatasetManagerTSERMamba
        else:
            train_fn = train_model
            eval_fn = evaluate_model
            dataset_class = DatasetManager

        for dataset_name in DATASETS:
            print(f"\n--- Dataset {dataset_name} ---")
            if model_name == 'ROCKET':
                print('dataset instantiation')
                datam = dataset_class(name=dataset_name, device=device, batch_size=BATCH_SIZE, is_transforming=False)
            else:
                datam = dataset_class(name=dataset_name, device=device, batch_size=BATCH_SIZE)
                train_loader, test_loader = datam.load_dataloader_for_training()

            if model_name == "TSERMamba":
                manager = ModelManager(model_name)
                model = manager.get_model(enc_in=datam.dims, seq_len=datam.seq_len)
                model.to(device)
            elif model_name == "ROCKET":
                print('creating rocket')
                model = RocketRegressor()

                print('fittando')
                start = time.time()
                model.fit(datam.X_train, datam.y_train)
                elapsed = time.time() - start

                print('predizeni')
                y_pred = model.predict(datam.X_test)
                metrics = calculate_metrics(datam.y_test, y_pred)
                wrap_results(metrics, model_name, dataset_name, elapsed)

                continue
            else:
                # Infer first batch for input size.
                first_batch = next(iter(train_loader))
                
                # Instantiate model dynamically.
                manager = ModelManager(model_name)
                model = manager.get_model(first_batch)
                model.to(device)

            # Get training hyperparameters from config.
            training_params = get_training_params(config, model.parameters())
            criterion = training_params["criterion"]
            optimizer = training_params["optimizer"]
            num_epochs = training_params["num_epochs"]
            patience = training_params["patience"]

            # Train the model.
            start = time.time()
            trained_model, best_loss, train_losses = train_fn(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                batch_size=BATCH_SIZE,
                num_epochs=num_epochs,
                patience=patience,
                batch_divisions=BATCH_DIVISIONS
            )
            elapsed = time.time() - start

            # Save training loss history.
            os.makedirs(f'{RESULTS_DIR}losses/{model_name}', exist_ok=True)
            with open(f'{RESULTS_DIR}losses/{model_name}/{model_name}_{dataset_name}.json', 'w') as f:
                json.dump(train_losses, f)

            # Free training memory.
            del optimizer, train_loader, train_losses
            gc.collect()
            torch.cuda.empty_cache()

            # Evaluate model.
            metrics = eval_fn(trained_model, test_loader, criterion, device, batch_divisions=BATCH_DIVISIONS)

            wrap_results(metrics, model_name, dataset_name, elapsed, best_loss)

            # Clean memory.
            del trained_model, test_loader
            gc.collect()
            torch.cuda.empty_cache()

    print("\n✅ All experiments completed successfully.")

if __name__ == "__main__":
    main()