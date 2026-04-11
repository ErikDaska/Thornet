import json
import logging
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from training.models import get_model
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import os
from datetime import datetime

# Switched to Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from datasets.tornet_dataset import TornetDataset
import pandas as pd

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

    Args:
        alpha (float):
            A standard weighting factor applied to the focal penalty. It acts as a
            baseline multiplier for the overall focal weight. (Default: 0.75)

        gamma (float):
            The "focusing" parameter. This is the core of Focal Loss.
            - If gamma = 0, this acts exactly like standard BCE.
            - As gamma increases (e.g., 2.0 or higher), the loss for "easy"
              predictions (like the massive amounts of empty, non-tornado space
              the model is highly confident is a 0) gets exponentially pushed toward 0.
            - This forces the model's gradients to exclusively update based on the
              "hard" examples (the actual, rare tornadoes).

        pos_weight (torch.Tensor):
            Passed directly into the underlying BCEWithLogitsLoss. This dictates the
            base ratio of importance between the positive and negative class.
            For extreme imbalance, this is usually set to (Num_Negatives / Num_Positives).
            If a tornado represents 1 in 100,000 pixels, pos_weight violently scales
            up the gradient of that single pixel so the model can't ignore it.
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
    logger.info(f"--- Starting {cfg.model.name} Training ---")

    # Paths and Configs
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv" # Fixed to raw_path to match evaluate script
    
    epochs = cfg.model.get("epochs", 5)
    batch_size = cfg.model.get("batch_size", 16)
    learning_rate = cfg.model.get("learning_rate", 1e-3)
    seed = cfg.model.get("seed", 42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    if not processed_data_path.exists():
        logger.error(f"Processed data not found at {processed_data_path}.")
        return

    dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)

    # --- 70/15/15 Stratified Split ---
    logger.info("Performing 70/15/15 stratified train/val/test split...")
    
    # Safely extract labels to stratify
    all_indices = list(range(len(dataset)))
    # Note: Depending on your dataset implementation, accessing dataset[i] might be slow. 
    # If TornetDataset has a fast way to get labels, replace this list comprehension.
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]

    # Split 1: 70% Train, 30% Temp 
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        all_indices, all_labels,
        test_size=0.30,
        stratify=all_labels,
        random_state=seed
    )

    # Split 2: Split the 30% Temp into 15% Val and 15% Test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed
    )

    logger.info(f"Final Split sizes -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # --- Subsets & DataLoaders ---
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Save Test Indices for Evaluation Task ---
    test_indices_path = Path(cfg.paths.processed_data_dir) / f"test_indices_{cfg.api.dataset.target_year}.csv"
    pd.DataFrame({"test_index": test_idx}).to_csv(test_indices_path, index=False)
    logger.info(f"Saved test indices to {test_indices_path}")

    # Initialize Model, Loss, and Optimizer
    model_kwargs = dict(cfg.model.get("params", {}))
    model_kwargs.setdefault("in_channels", len(dataset.variables))
    model = get_model(cfg.model.name, **model_kwargs).to(device)

    # --- HEAVY LOSS PENALIZER SETUP ---
    # Calculate the ratio of negative to positive samples to heavily weight the rare tornadoes
    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    calculated_pos_weight = torch.tensor([num_neg / float(num_pos)], dtype=torch.float32).to(device)
    logger.info(f"Calculated pos_weight for Focal Loss: {calculated_pos_weight.item():.2f}")

    criterion = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0, pos_weight=calculated_pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Configure MLflow
    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)
    mlflow.pytorch.autolog(log_models=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.model.name}_training_{cfg.api.dataset.target_year}_{current_time}"
    
    # Cleaned up history dictionary
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": cfg.model.name,
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

                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

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

            # --- LOGGING ---
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
            
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
            # Save to history dictionary
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_accuracy)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_accuracy)

        # --- GENERATE PLOTLY DASHBOARD ---
        logger.info("Generating and logging interactive Plotly training graphs...")
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model Accuracy'))
        x_epochs = list(range(1, epochs + 1))

        # Loss Plot
        fig.add_trace(go.Scatter(x=x_epochs, y=history['train_loss'], mode='lines+markers', name='Train Loss', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_epochs, y=history['val_loss'], mode='lines+markers', name='Val Loss', line=dict(color='orange')), row=1, col=1)

        # Accuracy Plot
        fig.add_trace(go.Scatter(x=x_epochs, y=history['train_acc'], mode='lines+markers', name='Train Accuracy', line=dict(color='green')), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_epochs, y=history['val_acc'], mode='lines+markers', name='Val Accuracy', line=dict(color='red')), row=1, col=2)

        fig.update_layout(
            title=f'{cfg.model.name} Training Metrics',
            xaxis_title='Epoch',
            xaxis2_title='Epoch',
            yaxis_title='Loss',
            yaxis2_title='Accuracy',
            template='plotly_white'
        )

        logger.info("Training complete. Saving model manually to MLflow...")

        # 1. Infer the model signature (Highly recommended for mixed 2D/3D architectures)
        # We grab one batch from the validation loader just to get the exact input/output shapes
        sample_input, _ = next(iter(val_loader))
        sample_input = sample_input.to(device)
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input)
            
        signature = infer_signature(sample_input.cpu().numpy(), sample_output.cpu().numpy())

        # 2. Log  the model
        registry_name = f"Tornet-{cfg.model.name}"
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
        )
        mlflow.set_tag("architecture", registry_name)

        # Log Plotly HTML and training history JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save Plotly Fig
            metrics_plot_path = os.path.join(tmpdir, "training_metrics.html")
            fig.write_html(metrics_plot_path)
            mlflow.log_artifact(metrics_plot_path, artifact_path="training_plots")

            # Save JSON history
            history_path = os.path.join(tmpdir, "training_history.json")
            with open(history_path, "w", encoding="utf-8") as history_file:
                json.dump(history, history_file, indent=2)
            mlflow.log_artifact(history_path, artifact_path="training_history")
        
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    train()