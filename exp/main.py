import os
import sys
import torch

# Add BASE folder (parent of exp/) to sys.path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Parser import Parser
from DatasetManager import DatasetManager
from models.ModelManager import ModelManager

def main():

    # Parse args and load config.
    parser = Parser()
    args = parser.parse_args()
    config = parser.load_config(args.config)

    # Extract from config.yaml.
    DATASETS = config.get("datasets", [])
    MODELS = config.get("models", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in DATASETS:
        
        print(f"\n=== Dataset {dataset_name} ===")
        datam = DatasetManager(name=dataset_name, device=device)
        train_loader, test_loader = datam.load_dataloader_for_training()

        # Infer input channels dynamically.
        first_batch = next(iter(train_loader))

        for model_name in MODELS:

            manager = ModelManager(model_name)
            model = manager.get_model(first_batch)  # input size inferred automatically.
            
            print(model)

if __name__ == "__main__":
    main()