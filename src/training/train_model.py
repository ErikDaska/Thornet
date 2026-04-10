import json
import logging
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from training.models import get_model

import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import os
from datetime import datetime

from torch.utils.data import DataLoader, random_split
from datasets.tornet_dataset import TornetDataset

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_loader(model: torch.nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float("nan")
    accuracy = correct_predictions / total_samples if total_samples > 0 else float("nan")
    return avg_loss, accuracy


# --- MAIN TRAINING LOOP ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def train(cfg: DictConfig):
    logger.info(f"--- Starting {cfg.model.name} Training ---")

    # Paths and Configs
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv"
    
    epochs = cfg.model.get("epochs", 5)
    batch_size = cfg.model.get("batch_size", 16)
    learning_rate = cfg.model.get("learning_rate", 1e-3)
    validation_split = cfg.model.get("validation_split", 0.0)
    seed = cfg.model.get("seed", 42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    if not processed_data_path.exists():
        logger.error(f"Processed data not found at {processed_data_path}.")
        return

    dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)
    if validation_split > 0.0:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        if val_size <= 0 or train_size <= 0:
            logger.error("validation_split must produce at least one sample for both train and validation sets.")
            return

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logger.info(f"Using validation split: {validation_split:.2f} ({train_size} train / {val_size} validation samples)")
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = None

    # Initialize Model, Loss, and Optimizer
    model_kwargs = dict(cfg.model.get("params", {}))
    model_kwargs.setdefault("in_channels", len(dataset.variables))
    model = get_model(cfg.model.name, **model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Configure MLflow
    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)

    mlflow.pytorch.autolog(log_models=True)


    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.model.name}_training_{cfg.api.dataset.target_year}_{current_time}"
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": cfg.model.name,
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
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_samples
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f} | Train Accuracy: {epoch_accuracy:.4f}")
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)
            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(epoch_accuracy)

            if val_loader is not None:
                val_loss, val_accuracy = evaluate_loader(model, val_loader, criterion, device)
                logger.info(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
            else:
                history["val_loss"].append(None)
                history["val_accuracy"].append(None)

        # Save and log the final model
        logger.info("Training complete. Saving model manually to MLflow...")
        mlflow.pytorch.log_model(model, "model")

        # Log training history artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "training_history.json")
            with open(history_path, "w", encoding="utf-8") as history_file:
                json.dump(history, history_file, indent=2)
            mlflow.log_artifact(history_path, artifact_path="training_history")
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    train()