import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 1. DATASET DEFINITION ---
class TornetDataset(Dataset):
    def __init__(self, data_dir: Path):
        """
        Expects a directory of .pt files where each file contains:
        a dictionary or tuple with {'features': tensor_3d, 'label': 0_or_1}
        """
        self.files = sorted(list(data_dir.rglob("*.pt")))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # NOTE: This assumes you update data_processing.py to save both data AND labels
        # e.g., torch.save({'features': tensor, 'label': label}, out_path)
        data = torch.load(self.files[idx])
        
        # Fallback dummy logic if labels aren't implemented yet (for testing the pipeline)
        if isinstance(data, torch.Tensor):
            features = data
            label = torch.tensor(0.0) # Dummy label
        else:
            features = data['features']
            label = torch.tensor(float(data['label']))
            
        return features, label


# --- 2. 3D CNN MODEL ---
class Tornet3DCNN(nn.Module):
    def __init__(self, in_channels=6):
        super(Tornet3DCNN, self).__init__()
        
        # Input shape expected: (Batch, Channels, Sweeps, Azimuth, Range)
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling reduces spatial/depth dimensions to 1x1x1
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layer for binary classification (Tornado vs None)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- 3. MAIN TRAINING LOOP ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def train(cfg: DictConfig):
    logger.info("--- Starting 3D CNN Model Training ---")

    # Paths and Configs
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    epochs = cfg.training.get("epochs", 5)
    batch_size = cfg.training.get("batch_size", 16)
    learning_rate = cfg.training.get("learning_rate", 1e-3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    if not processed_data_path.exists():
        logger.error(f"Processed data not found at {processed_data_path}.")
        return

    dataset = TornetDataset(processed_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize Model, Loss, and Optimizer
    model = Tornet3DCNN(in_channels=6).to(device)
    criterion = nn.BCEWithLogitsLoss() # Best for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Configure MLflow
    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)

    run_name = f"3dcnn_training_{cfg.api.dataset.target_year}"
    
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": "3D_CNN",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss"
        })

        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
            
            # Log metrics per epoch
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Save and log the final model
        logger.info("Training complete. Saving model to MLflow...")
        mlflow.pytorch.log_model(model, "tornet_3dcnn_model")
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    train()