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
import xarray as xr
import numpy as np
import pandas as pd
import os

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- 1. DATASET DEFINITION (HYPER-OPTIMIZED) ---
class TornetDataset(Dataset):
    def __init__(self, data_dir: Path, catalog_path: Path, variables=None):
        self.files = sorted(list(data_dir.rglob("*.nc")))

        if variables is None:
            self.variables = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'MASK']
        else:
            self.variables = variables

        if not self.files:
            logger.warning(f"No .nc files found in {data_dir}!")

        # --- 1. Load Labels from Catalog ---
        logger.info(f"Loading labels from catalog: {catalog_path}")
        if catalog_path.exists():
            self.catalog = pd.read_csv(catalog_path)
        else:
            logger.error(f"Catalog not found at {catalog_path}. Labels will default to 0.0!")
            self.catalog = pd.DataFrame()

        # --- 2. PRE-COMPUTE LABELS (Vectorized - Lightning Fast) ---
        self.label_map = {}
        logger.info("Pre-computing label mapping...")
        if not self.catalog.empty:
            # Filter the catalog ONCE to get only tornadoes
            tor_catalog = self.catalog[self.catalog['category'] == 'TOR']

            # Combine all columns into a single giant string block for instant searching
            tor_text_blob = tor_catalog.to_string()

            for f in self.files:
                original_filename = f.name.replace("processed_", "")
                # Instant check: Is this filename anywhere inside the Tornado text block?
                if original_filename in tor_text_blob:
                    self.label_map[str(f)] = 1.0
                else:
                    self.label_map[str(f)] = 0.0

        # --- 3. Build an Index Map (Skipping xarray open!) ---
        self.index_map = []
        logger.info("Building dataset index...")
        for f in self.files:
            # We assume 1 time step per file (idx 0) to bypass slow NetCDF reading
            self.index_map.append((f, 0))

        logger.info(f"Dataset mapped: {len(self.index_map)} total 3D scans available.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, time_idx = self.index_map[idx]

        # We only open the file right when we need it for the batch
        with xr.open_dataset(file_path, engine="netcdf4") as ds:
            if 'time' in ds.dims:
                step_ds = ds.isel(time=time_idx)
            else:
                step_ds = ds

            channels = []
            for var in self.variables:
                if var in step_ds.data_vars:
                    data = step_ds[var].values
                else:
                    shape = (step_ds.sizes.get('sweep', 1),
                             step_ds.sizes.get('azimuth', 1),
                             step_ds.sizes.get('range', 1))
                    data = np.zeros(shape, dtype=np.float32)

                channels.append(data)

            features = np.stack(channels, axis=0)
            features_tensor = torch.from_numpy(features).float()

        label = self.label_map.get(str(file_path), 0.0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features_tensor, label_tensor

# --- 2. 3D CNN MODEL ---
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


# --- 3. MAIN TRAINING LOOP ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def train(cfg: DictConfig):
    logger.info("--- Starting 3D CNN Model Training ---")

    # Paths and Configs
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv"
    
    epochs = cfg.training.get("epochs", 5)
    batch_size = cfg.training.get("batch_size", 16)
    learning_rate = cfg.training.get("learning_rate", 1e-3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    if not processed_data_path.exists():
        logger.error(f"Processed data not found at {processed_data_path}.")
        return

    dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialize Model, Loss, and Optimizer
    model = Tornet3DCNN(in_channels=7).to(device)
    criterion = torch.nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Configure MLflow
    mlflow.set_registry_uri("sqlite:////mlflow/mlflow.db")
    mlflow.set_tracking_uri(cfg.tracking.uri)
    os.environ['MLFLOW_ARTIFACT_ROOT'] = "/mlruns"
    mlflow.set_experiment(cfg.tracking.experiment_name)

    mlflow.pytorch.autolog(log_models=True)

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
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # --- NEW: Calculate Accuracy ---
                # 1. Convert logits to probabilities using sigmoid
                probs = torch.sigmoid(outputs)
                # 2. Threshold at 0.5 to get binary prediction (1 or 0)
                predictions = (probs > 0.5).float()
                # 3. Count how many matched the true labels
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

            # Calculate epoch averages
            avg_loss = running_loss / len(dataloader)
            epoch_accuracy = correct_predictions / total_samples
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Accuracy: {epoch_accuracy:.4f}")
            
            # --- NEW: Log metrics to MLflow per epoch ---
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)

        # Save and log the final model
        logger.info("Training complete. Saving model manually to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    train()