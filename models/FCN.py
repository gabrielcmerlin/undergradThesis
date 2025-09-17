import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, input_size: int):
        """
        Fully Convolutional Network for regression with 1 output.
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Second convolutional block
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Third convolutional block
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Final linear layer for regression (single output)
        self.final_layer = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x