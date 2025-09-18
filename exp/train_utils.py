import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, criterion, optimizer, device, 
                num_epochs=100, patience=10, print_every=1):
    """
    Train a PyTorch model with early stopping based on training loss.
    """

    model.train()  # Set model to training mode
    best_loss = float('inf')  # Initialize best loss
    train_losses = []  # Store training loss per epoch
    best_model_state = None  # Store best model weights
    epochs_no_improve = 0  # Counter for early stopping

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # Move data to device

            optimizer.zero_grad()  # Reset gradients
            outputs = model(X)  # Forward pass
            loss = criterion(outputs, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item() * X.size(0)  # Accumulate batch loss

        # Compute average loss for the epoch.
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # Print training progress.
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.6f}")

        # Check if the model improved.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Apply early stopping.
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best training loss: {best_loss:.6f}")
            break

    # Load the best model weights.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_loss, train_losses


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a PyTorch model on test data and compute regression metrics.
    """

    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    all_preds, all_labels = [], []
    with torch.no_grad():  # Disable gradient computation
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)

            # Collect predictions and true labels.
            preds = outputs.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy().flatten())

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\nTest Loss: {avg_loss:.6f}")
    
    # Compute regression metrics.
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    rmse = np.sqrt(mse)

    return {
        'mse': mse, 
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'y_true': all_labels,
        'y_pred': all_preds
    }

def scatter_ytrue_ypred(y_true, y_pred, title, save_path=None):
    """
    Plot a scatter plot comparing true vs predicted values.
    """

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)  # Scatter points

    # Plot ideal diagonal line (y = x).
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal (y = x)')
    plt.xlabel("Valores reais (y_true)")
    plt.ylabel("Predições (y_pred)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()