import torch
import os

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    """
    Train the model and return it after training.
    """
    
    model.train()

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.6f}")

    return model

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    """

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"\nTest Loss: {avg_loss:.6f}\n")
    
    return avg_loss