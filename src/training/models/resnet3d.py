import torch
import torch.nn as nn
import torchvision.models.video as video_models

class TornetResNet3D(nn.Module):
    def __init__(self, in_channels=6):
        super(TornetResNet3D, self).__init__()
        # Load a 3D ResNet-18 (untrained, as RGB weights won't map well to radar data)
        self.model = video_models.r3d_18(weights=None)
        
        # Swap the first conv layer to accept 6 channels
        original_conv = self.model.stem[0]
        self.model.stem[0] = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Swap the final classifier layer for binary classification (Tornado vs. No Tornado)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        return self.model(x)