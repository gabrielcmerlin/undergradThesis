import torch
import torch.nn as nn
import torch.optim as optim

def get_training_params(config, model_params=None):
    """
    Extracts all training parameters from config and returns
    criterion, optimizer, learning rate, betas, eps, and num_epochs.
    
    Args:
        config (dict): Dictionary loaded from YAML.
        model_params (dict, optional): Extra params needed for optimizer (like model.parameters())
    
    Returns:
        dict: Contains 'criterion', 'optimizer', 'num_epochs'
    """

    # Loss function.
    criterion_name = config.get("criterion", "MSELoss")
    criterion_map = {
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "CrossEntropyLoss": nn.CrossEntropyLoss
    }
    criterion = criterion_map.get(criterion_name, nn.MSELoss)()

    # Optimizer.
    optimizer_name = config.get("optimizer", "Adam")
    lr = config.get("lr", 0.001)
    betas = tuple(config.get("betas", (0.9, 0.999)))
    eps = float(config.get("eps", 1e-8))

    optimizer = None
    if model_params is not None:
        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(model_params, lr=lr, betas=betas, eps=eps)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    num_epochs = config.get("num_epochs", 100)
    patience = config.get("patience", 10)

    return {
        "criterion": criterion,
        "optimizer": optimizer,
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "num_epochs": num_epochs,
        "patience": patience
    }