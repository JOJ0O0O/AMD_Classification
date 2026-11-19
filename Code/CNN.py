import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



class CNN(nn.Module):
    """Eq. (10.arch) with two conv blocks; 28 -> 14 -> 7 by Eq. (10.outsize)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 'same': 28 x 28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 'same': 14 x 14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # halves each time
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x))) # (N, 64, 7, 7)
        x = self.pool(F.relu(self.conv3(x))) # (N, 64, 7, 7)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x) # logits, not probabilities


