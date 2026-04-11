import torch
import torch.nn as nn

class SpatialCNN_GRU(nn.Module):
    def __init__(self, in_channels=6, cnn_features=64, gru_hidden=32):
        super(SpatialCNN_GRU, self).__init__()

        # 1. Spatial Feature Extractor (2D CNN)
        # We use 2D convolutions to process each spatial slice independently.
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # Reduce height only, preserve width when width may be 1

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # Continue reducing height only
            
            nn.Conv2d(32, cnn_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_features),
            nn.ReLU(),
            # Adaptive pool squashes the remaining spatial dims to 1x1
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        self.feature_dim = cnn_features
        
        # 2. Sequence Model (GRU)
        # Processes the sequence of spatial features extracted by the CNN
        self.gru = nn.GRU(
            input_size=self.feature_dim, 
            hidden_size=gru_hidden, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Expected input x: (Batch, Channels, Depth, Height, Width)
        # e.g., (B, 6, D, 120, 120)
        B, C, D, H, W = x.shape
        
        # --- STEP 1: Reshape for the 2D CNN ---
        # Swap Channels and Depth: (B, D, C, H, W)
        x = x.transpose(1, 2).contiguous() 
        
        # Merge Batch and Depth: (B * D, C, H, W)
        # Now every single slice is treated as a separate image in a massive batch
        x = x.view(B * D, C, H, W)
        
        # --- STEP 2: Extract Spatial Features ---
        cnn_out = self.spatial_cnn(x)           # Output: (B * D, cnn_features, 1, 1)
        cnn_out = torch.flatten(cnn_out, 1)     # Output: (B * D, cnn_features)
        
        # --- STEP 3: Reshape for the GRU ---
        # Un-merge Batch and Depth to form sequences: (B, D, cnn_features)
        rnn_in = cnn_out.view(B, D, self.feature_dim)
        
        # --- STEP 4: Sequence Processing ---
        # gru_out contains the output features for each step in the sequence
        gru_out, hidden = self.gru(rnn_in)
        
        # Grab the output from the final step in the sequence to classify the whole volume
        last_out = gru_out[:, -1, :]            # Output: (B, gru_hidden)
        
        # --- STEP 5: Classify ---
        out = self.classifier(last_out)         # Output: (B, 1)
        return out