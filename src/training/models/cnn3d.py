import torch
import torch.nn as nn


class Tornet3DCNN(torch.nn.Module):
    def __init__(self, in_channels=6):
        super(Tornet3DCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            # CHANGED: Pool the first two dims (120x120), leave the last dim (1) alone
            torch.nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)), 
            
            torch.nn.Conv3d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            # CHANGED: Pool the first two dims, leave the last dim alone
            torch.nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        )
        self.global_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x