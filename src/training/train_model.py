import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.tornet_dataset import TornetDataset
import xarray as xr
import numpy as np
import pandas as pd
import random
from torch.utils.data import Subset

from models.CNN import Tornet3DCNN, Tornet2DCNN

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss designed to prevent Mode Collapse in heavily imbalanced spatial maps.
    It dynamically scales the loss based on the model's confidence, forcing it to
    pay attention to the rare positive (tornado) cases instead of spamming '0'.
    """

    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Calculate standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )

        # Calculate probabilities to determine confidence
        probs = torch.sigmoid(inputs)

        # pt is the predicted probability for the TRUE class
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Apply the focal multiplier: (1 - pt)^gamma
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        return (focal_weight * bce_loss).mean()

# --- MAIN TRAINING LOOP ---
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

    full_dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)

    logger.info("Extracting aligned labels from dataset...")
    all_y = np.array([full_dataset.label_map[str(f)] for f, _ in full_dataset.index_map])
    all_indices = list(range(len(full_dataset)))

    data_fraction = cfg.training.get("data_fraction", 1.0)
    if data_fraction < 1.0:
        # We use train_test_split to safely "throw away" the unused fraction
        # while perfectly maintaining the class ratio in the kept data.
        used_idx, _, used_y, _ = train_test_split(
            all_indices, all_y,
            train_size=data_fraction,
            stratify=all_y,
            random_state=42
        )
        dataset = Subset(full_dataset, used_idx)
        logger.info(
            f"Stratified Subsampling: Using {data_fraction * 100}% of data ({len(used_idx)}/{len(all_indices)} scans).")
    else:
        used_idx = all_indices
        used_y = all_y
        dataset = full_dataset
        logger.info(f"Using full dataset: {len(dataset)} scans.")

    # --- 70/15/15 Stratified Split (Safely Aligned) ---
    logger.info("Performing 70/15/15 stratified train/val/test split...")

    # Split 1: 70% Train, 30% Temp (using the data we decided to keep)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        used_idx, used_y,
        test_size=0.30,
        stratify=used_y,
        random_state=42
    )

    # Split 2: Split the 30% Temp into 15% Val and 15% Test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    logger.info(f"Final Split sizes -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # --- Subsets & DataLoaders ---
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Save Test Indices for Evaluation Task ---
    test_indices_path = Path(cfg.paths.processed_data_dir) / f"test_indices_{cfg.api.dataset.target_year}.csv"
    pd.DataFrame({"test_index": test_idx}).to_csv(test_indices_path, index=False)
    logger.info(f"Saved test indices to {test_indices_path}")

    # Initialize Model, Loss, and Optimizer
    model = Tornet2DCNN(in_channels=14).to(device)

    # Calculate how many negative vs positive samples are in the training set
    num_positives = np.sum(y_train == 1.0)
    num_negatives = np.sum(y_train == 0.0)

    # Weight = (Number of Negatives) / (Number of Positives)
    pos_weight_val = num_negatives / num_positives if num_positives > 0 else 1.0
    logger.info(f"Training Class Ratio -> Neg: {num_negatives}, Pos: {num_positives}. Applying pos_weight={pos_weight_val:.2f}")

    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0, pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Configure MLflow
    mlflow.set_registry_uri("sqlite:////mlflow/mlflow.db")
    mlflow.set_tracking_uri(cfg.tracking.uri)
    os.environ['MLFLOW_ARTIFACT_ROOT'] = "/mlruns"
    mlflow.set_experiment(cfg.tracking.experiment_name)
    mlflow.pytorch.autolog(log_models=True)

    run_name = f"3dcnn_training_{cfg.api.dataset.target_year}"

    # Track metrics for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": "3D_CNN",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx)
        })

        # Training Loop
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                # --- Calculate Accuracy ---
                # 1. Convert logits to probabilities using sigmoid
                probs = torch.sigmoid(outputs)
                # 2. Threshold at 0.5 to get binary prediction (1 or 0)
                predictions = (probs > 0.5).float()
                # 3. Count how many matched the true labels
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

            # Calculate epoch averages
            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_predictions / total_samples

            # --- VALIDATION PHASE ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    predictions = (probs > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total

            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # --- Log metrics to MLflow per epoch ---
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(val_accuracy)

        # --- GENERATE PLOTS ---
        logger.info("Generating and logging training plots...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy', marker='o')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        mlflow.log_figure(fig, "metrics_plot.png")
        plt.close(fig)

        logger.info("Training complete. Saving model manually to MLflow...")
        mlflow.pytorch.log_model(model, "model")
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    train()