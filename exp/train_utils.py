import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, criterion, optimizer, device, batch_size=None,
                num_epochs=100, patience=10, print_every=1, batch_divisions=1):
    """
    Train a PyTorch model with optional batch splitting (gradient accumulation).
    Ensures equivalence with standard batch training when batch_divisions=1.
    """

    model.train()
    best_loss = float('inf')
    train_losses = []
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):

        total_loss = 0.0
        torch.cuda.empty_cache()

        for X, y in train_loader:

            actual_batch_size = X.size(0)

            # Determine samples per division
            if batch_divisions > 1:
                samples_per_div = int(np.ceil(actual_batch_size / batch_divisions))
            else:
                samples_per_div = actual_batch_size  # equivalent to no division

            optimizer.zero_grad()  # zero grad once per batch
            batch_loss = 0.0

            for i in range(batch_divisions):
                start = i * samples_per_div
                end = min((i + 1) * samples_per_div, actual_batch_size)

                if start >= end:  # skip empty slice
                    continue

                X_div = X[start:end]
                y_div = y[start:end]
                X_div, y_div = X_div.to(device), y_div.to(device)

                # Forward pass
                outputs = model(X_div)
                loss = criterion(outputs, y_div)

                # Scale loss to account for batch division
                loss = loss * (X_div.size(0) / actual_batch_size)

                # Backward
                loss.backward()

                batch_loss += loss.item() * actual_batch_size / X_div.size(0)  # reverse scaling for logging

            optimizer.step()  # update weights after full batch
            total_loss += batch_loss

        # Average loss per epoch
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best training loss: {best_loss:.6f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_loss, train_losses

def evaluate_model(model, test_loader, criterion, device, batch_divisions=1):
    """
    Evaluate a PyTorch model on test data with optional batch divisions to avoid OOM.
    Computes regression metrics (MSE, MAE, R2, RMSE).
    """

    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            actual_batch_size = X.size(0)
            
            if batch_divisions > 1:
                samples_per_div = int(np.ceil(actual_batch_size / batch_divisions))
            else:
                samples_per_div = actual_batch_size

            batch_preds = []
            batch_loss = 0.0

            for i in range(batch_divisions):
                start = i * samples_per_div
                end = min((i + 1) * samples_per_div, actual_batch_size)
                if start >= end:
                    continue

                X_div = X[start:end].to(device)
                y_div = y[start:end].to(device)

                outputs = model(X_div)
                loss = criterion(outputs, y_div)
                # Scale loss as in training
                loss = loss * (X_div.size(0) / actual_batch_size)
                batch_loss += loss.item() * actual_batch_size / X_div.size(0)

                batch_preds.append(outputs.cpu())

            # Concat batch predictions
            batch_preds = torch.cat(batch_preds, dim=0)
            all_preds.extend(batch_preds.numpy().flatten())
            all_labels.extend(y.numpy().flatten())
            total_loss += batch_loss

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\nTest Loss: {avg_loss:.6f}")

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
    Plot scatter of true vs predicted values.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
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

def train_TSERMamba(model, train_loader, criterion, optimizer, device, batch_size=None,
                    num_epochs=100, patience=10, print_every=1, batch_divisions=1):
    """
    Train TSERMamba regression model using two inputs (CWT + raw/another features),
    following the thesis training template and TSCMamba structure.
    """
    model.train()
    best_loss = float('inf')
    train_losses = []
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        torch.cuda.empty_cache()

        for batch_x_cwt, batch_x_another, y in train_loader:
            actual_batch_size = batch_x_cwt.size(0)
            samples_per_div = int(np.ceil(actual_batch_size / batch_divisions)) if batch_divisions > 1 else actual_batch_size

            optimizer.zero_grad()
            batch_loss = 0.0

            for i in range(batch_divisions):
                start = i * samples_per_div
                end = min((i + 1) * samples_per_div, actual_batch_size)
                if start >= end:
                    continue

                x_cwt = batch_x_cwt[start:end].to(device).float()
                x_another = batch_x_another[start:end].to(device).float()
                y_div = y[start:end].to(device).float()

                outputs = model(x_cwt, x_another)
                loss = criterion(outputs, y_div)
                loss = loss * (x_cwt.size(0) / actual_batch_size)
                loss.backward()

                batch_loss += loss.item() * actual_batch_size / x_cwt.size(0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss

        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best training loss: {best_loss:.6f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_loss, train_losses

def evaluate_TSERMamba(model, test_loader, criterion, device, batch_divisions=1):
    """
    Evaluate TSERMamba model on regression test data (CWT + another input).
    Computes regression metrics (MSE, MAE, R2, RMSE).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_x_cwt, batch_x_another, y in test_loader:
            actual_batch_size = batch_x_cwt.size(0)
            samples_per_div = int(np.ceil(actual_batch_size / batch_divisions)) if batch_divisions > 1 else actual_batch_size

            batch_preds = []
            batch_loss = 0.0

            for i in range(batch_divisions):
                start = i * samples_per_div
                end = min((i + 1) * samples_per_div, actual_batch_size)
                if start >= end:
                    continue

                x_cwt = batch_x_cwt[start:end].to(device).float()
                x_another = batch_x_another[start:end].to(device).float()
                y_div = y[start:end].to(device).float()

                outputs = model(x_cwt, x_another)
                loss = criterion(outputs, y_div)
                loss = loss * (x_cwt.size(0) / actual_batch_size)
                batch_loss += loss.item() * actual_batch_size / x_cwt.size(0)

                batch_preds.append(outputs.cpu())

            batch_preds = torch.cat(batch_preds, dim=0)
            all_preds.extend(batch_preds.numpy().flatten())
            all_labels.extend(y.numpy().flatten())
            total_loss += batch_loss

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\nTest Loss: {avg_loss:.6f}")

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