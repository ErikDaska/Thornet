import torch
from torch import nn

class Tornet2DCNN(nn.Module):
    def __init__(self, in_channels=7, num_sweeps=2, **kwargs):
        super(Tornet2DCNN, self).__init__()

        flattened_channels = in_channels * num_sweeps

        # --- BLOCK 1 (Starts with 32 filters) ---
        self.block1 = nn.Sequential(
            nn.Conv2d(flattened_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        # --- BLOCK 2 (Doubles to 64) ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        # --- BLOCK 3 (Doubles to 128) ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        # --- BLOCK 4 (Doubles to 256) ---
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        # --- TORNADO LIKELIHOOD MAP ---
        # Reduces the 256 channel dimension down to a single 2D heat map
        self.likelihood_map = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
            # Note: The paper uses a Sigmoid here. However, training script uses
            # BCEWithLogitsLoss, which automatically applies Sigmoid mathematically.
            # We output raw logits here to prevent double-applying it and ruining training!
        )

        # --- GLOBAL MAX POOLING ---
        # Finds the highest single probability pixel in the likelihood map
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # 1. FLATTEN 3D TO 2D
        if x.dim() == 5:
            # Actual shape from dataloader: [batch, channels, azimuth, range, sweeps]
            batch, channels, h, w, sweeps = x.shape

            # Step A: Rearrange dimensions to [batch, channels, sweeps, azimuth, range]
            x = x.permute(0, 1, 4, 2, 3)

            # Step B: Flatten the channels and sweeps together (7 * 2 = 14)
            # .contiguous() is required after permute before calling .view()
            x = x.contiguous().view(batch, channels * sweeps, h, w)

        # 2. FEATURE EXTRACTION
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 3. LIKELIHOOD MAPPING
        x = self.likelihood_map(x)

        # 4. GLOBAL MAX POOLING
        x = self.global_max_pool(x)

        # 5. FLATTEN FOR LOSS FUNCTION
        x = torch.flatten(x, 1)

        return x