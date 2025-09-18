import os
import sys
import time
import pandas as pd
import json

# Add BASE folder (parent of exp/) to sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import random
import numpy as np
import torch

# Fix random seeds for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from Parser import Parser
from DatasetManager import DatasetManager
from models.ModelManager import ModelManager
from training_config import get_training_params
from train_utils import train_model, evaluate_model, scatter_ytrue_ypred

EXPERIMENT_DATE = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = '../outputs/'

def main():

    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract datasets and models.
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    for model_name in MODELS:
        print(f"\n=== Model: {model_name} ===")

        results_data_dir = {
            'model': [], 'dataset': [], 'mse': [],
            'mae': [], 'r2': [], 'rmse': [],
            'best_train_loss': [], 'time': []
        }

        for dataset_name in DATASETS:
            print(f"\n--- Dataset {dataset_name} ---")
            datam = DatasetManager(name=dataset_name, device=device)
            train_loader, test_loader = datam.load_dataloader_for_training()

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
            trained_model, best_loss, train_losses = train_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
                patience=patience
            )
            elapsed = time.time() - start

            with open(f'{RESULTS_DIR}losses/{model_name}/{model_name}_{dataset_name}.json', 'w') as f:
                json.dump(train_losses, f)

            # Evaluate model.
            metrics = evaluate_model(trained_model, test_loader, criterion, device)

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

            pd.DataFrame(results_data_dir).to_csv(f"{RESULTS_DIR}results/{model_name}_{EXPERIMENT_DATE}.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()