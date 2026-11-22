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
        self.c1, self.c2, self.c3 = channels
        self.conv1 = nn.Conv2d(3, self.c1, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(self.c1, self.c2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(self.c2, self.c3, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # halves each time
        self.fc1 = nn.Linear(self.c3 * 4 * 4, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 4)
        self.dropout = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) #Inserted to be able to work adaptively with different sizes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), self.c3 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x) # logits, not probabilities