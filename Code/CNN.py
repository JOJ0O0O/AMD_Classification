import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



class CNN(nn.Module):
    """Taken from lecturebook with modifications to make channels, kernel size, dropout and number of hidden features an input"""
    def __init__(self, channels=(32,64,128), kernel_size=3, dropout=0.5, hidden_features=256):
        """
        channels: tuple, number of channels for every filter layer
        kernel_size: Size of the kernel for all convolutional layers
        dropout: float
        hidden_features: int, number of neurons in the FC layer
        """
        super().__init__()

        channels = tuple(channels)
        if not channels or any(c <= 0 for c in channels):
            raise ValueError("channels must contain one or more positive values")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")

        layers = []
        in_channels = 3

        padding = kernel_size // 2
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(channels[-1], hidden_features)
        self.fc2 = nn.Linear(hidden_features, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)  # logits