import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch

import json
import gc

from aeon.regression.convolution_based import MiniRocketRegressor
from train_utils import calculate_metrics, scatter_ytrue_ypred
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

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

# Note: seeding is done per run inside main() to allow multiple runs with varying seeds.
# Global deterministic flags are not forced here so model runs can vary.

def wrap_results(metrics, model_name, dataset_name, elapsed, best_loss=None, run_suffix=None):
    results_data_dir = {
            'model': [], 'dataset': [], 'mse': [],
            'mae': [], 'r2': [], 'rmse': [],
            'best_train_loss': [], 'time': [], 'run': []
    }
    
    # Scatter plot.
    os.makedirs(f'{RESULTS_DIR}scatter/{model_name}', exist_ok=True)
    scatter_ytrue_ypred(metrics['y_true'], metrics['y_pred'],
                title=f"{dataset_name}",
                save_path=f'{RESULTS_DIR}scatter/{model_name}/{model_name}_{dataset_name}{"_" + run_suffix if run_suffix else ""}.png')
 
    # Store results.
    results_data_dir['model'].append(model_name)
    results_data_dir['dataset'].append(dataset_name)
    results_data_dir['mse'].append(metrics['mse'])
    results_data_dir['mae'].append(metrics['mae'])
    results_data_dir['r2'].append(metrics['r2'])
    results_data_dir['rmse'].append(metrics['rmse'])
    results_data_dir['best_train_loss'].append(best_loss)
    results_data_dir['time'].append(elapsed)
    # store run identifier so all runs go into the same CSV but are distinguishable
    results_data_dir['run'].append(run_suffix if run_suffix else "")
 
    os.makedirs(f"{RESULTS_DIR}results", exist_ok=True)
    # Write all runs into the same CSV per model (timestamped by EXPERIMENT_DATE).
    results_path = f"{RESULTS_DIR}results/{model_name}_{EXPERIMENT_DATE}.csv"
    df = pd.DataFrame(results_data_dir)
    df.to_csv(
        results_path,
        mode='a',
        index=False,
        header=not os.path.exists(results_path)
)

def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Number of repetitions (runs) from config. Default 1.
    NUM_RUNS = int(config.get("runs", 1))
    BASE_SEED = config.get("seed", None)  # optional base seed; if provided, each run uses BASE_SEED + run_idx

    # Extract datasets and models.
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])
    BATCH_SIZE = config.get("batch_size", 16)
    BATCH_DIVISIONS = config.get("batch_divisions", 8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')
 
    for run_idx in range(NUM_RUNS):
        # determine seed for this run (if BASE_SEED provided, use BASE_SEED + run_idx; else randomize)
        if BASE_SEED is not None:
            seed = int(BASE_SEED) + run_idx
        else:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
 
        # apply per-run seeding (allows variability between runs)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # allow nondeterministic behavior for variability
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        run_suffix = f"{EXPERIMENT_DATE}_run{run_idx+1}_{seed}"
        print(f"\n=== Run {run_idx+1}/{NUM_RUNS} | seed={seed} ===")
 
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
 
                if model_name not in ['MiniROCKET', 'XGBoost', 'RandomForest']:
                    datam = dataset_class(name=dataset_name, device=device, batch_size=BATCH_SIZE)
                    train_loader, test_loader = datam.load_dataloader_for_training()
 
                if model_name == "TSERMamba":
                    manager = ModelManager(model_name)
                    model = manager.get_model(enc_in=datam.dims, seq_len=datam.seq_len)
                    model.to(device)
                elif model_name == "MiniROCKET":
                    datam = dataset_class(name=dataset_name, device=device, batch_size=BATCH_SIZE, is_transforming=False)
 
                    model = MiniRocketRegressor()
 
                    start = time.time()
                    model.fit(datam.X_train, datam.y_train)
                    elapsed = time.time() - start
 
                    y_pred = model.predict(datam.X_test)
                    metrics = calculate_metrics(datam.y_test, y_pred)
                    wrap_results(metrics, model_name, dataset_name, elapsed, run_suffix=run_suffix)
 
                    continue
                elif model_name == "XGBoost":
                    datam = dataset_class(name=dataset_name, device=device, batch_size=BATCH_SIZE, is_transforming=False)
 
                    # XGBoost using default parameters, only number of trees specified
                    model = XGBRegressor(n_estimators=100, random_state=seed, subsample=0.8, colsample_bytree=0.8)
 
                    start = time.time()
                    model.fit(datam.X_train.reshape(len(datam.X_train), -1), datam.y_train)
                    elapsed = time.time() - start
 
                    y_pred = model.predict(datam.X_test.reshape(len(datam.X_test), -1))
                    metrics = calculate_metrics(datam.y_test, y_pred)
                    wrap_results(metrics, model_name, dataset_name, elapsed, run_suffix=run_suffix)
                 
                    continue
                elif model_name == "RandomForest":
                    datam = dataset_class(name=dataset_name, device=device, batch_size=BATCH_SIZE, is_transforming=False)
 
                    model = RandomForestRegressor(n_estimators=100, random_state=seed)
 
                    start = time.time()
                    model.fit(datam.X_train.reshape(len(datam.X_train), -1), datam.y_train)
                    elapsed = time.time() - start
 
                    y_pred = model.predict(datam.X_test.reshape(len(datam.X_test), -1))
                    metrics = calculate_metrics(datam.y_test, y_pred)
                    wrap_results(metrics, model_name, dataset_name, elapsed, run_suffix=run_suffix)
                 
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
                with open(f'{RESULTS_DIR}losses/{model_name}/{model_name}_{dataset_name}{"_" + run_suffix if run_suffix else ""}.json', 'w') as f:
                    json.dump(train_losses, f)
 
                # Free training memory.
                del optimizer, train_loader, train_losses
                gc.collect()
                torch.cuda.empty_cache()
 
                # Evaluate model.
                metrics = eval_fn(trained_model, test_loader, criterion, device, batch_divisions=BATCH_DIVISIONS)
 
                wrap_results(metrics, model_name, dataset_name, elapsed, best_loss, run_suffix=run_suffix)
 
                # Clean memory.
                del trained_model, test_loader
                gc.collect()
                torch.cuda.empty_cache()
 
    print("\n✅ All experiments completed successfully.")
 
if __name__ == "__main__":
    main()