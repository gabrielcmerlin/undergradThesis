import os
import sys

# Add BASE folder (parent of exp/) to sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import random
import numpy as np
import torch

# Fix random seeds for reproducibility
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
from train_utils import train_model, evaluate_model

def main():

    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract datasets and models.
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nDevice: {device}')

    for dataset_name in DATASETS:
        print(f"\n=== Dataset {dataset_name} ===")
        datam = DatasetManager(name=dataset_name, device=device)
        train_loader, test_loader = datam.load_dataloader_for_training()

        # Infer first batch for input size.
        first_batch = next(iter(train_loader))

        for model_name in MODELS:
            print(f"\n--- Model: {model_name} ---")

            # Instantiate model dynamically.
            manager = ModelManager(model_name)
            model = manager.get_model(first_batch)
            model.to(device)

            # Get training hyperparameters from config.
            training_params = get_training_params(config, model.parameters())
            criterion = training_params["criterion"]
            optimizer = training_params["optimizer"]
            num_epochs = training_params["num_epochs"]

            # Train the model.
            trained_model = train_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
            )

            # Evaluate model.
            evaluate_model(trained_model, test_loader, criterion, device)

if __name__ == "__main__":
    main()