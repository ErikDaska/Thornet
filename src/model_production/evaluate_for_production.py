import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score

import hydra
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient
from datasets.tornet_dataset import TornetDataset
from training.models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def evaluate_for_production(cfg: DictConfig):
    model_name = cfg.model.name
    registry_name = f"Tornet-{model_name}"
    logger.info(f"Starting Production Evaluation for: {registry_name}")
    
    mlflow.set_tracking_uri(cfg.tracking.uri)
    client = MlflowClient()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get the latest version of this specific architecture
    try:
        latest_version = client.get_latest_versions(name=registry_name)[0]
    except Exception as e:
        logger.warning(f"No registered model found for {registry_name}. Skipping evaluation.")
        return

    model_uri = f"models:/{registry_name}/{latest_version.version}"
    
    # 2. Load Dataset
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv"
    test_indices_path = Path(cfg.paths.processed_data_dir) / f"test_indices_{cfg.api.dataset.target_year}.csv"

    full_dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)
    test_indices = pd.read_csv(test_indices_path)['test_index'].tolist()
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 3. Evaluate Model
    logger.info(f"Evaluating {registry_name} (Version {latest_version.version})...")
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()

    y_true, y_scores = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_scores.extend(probs)
            y_true.extend(labels.numpy())

    fresh_ap = average_precision_score(np.array(y_true), np.array(y_scores))
    logger.info(f"Calculated Fresh AP Score: {fresh_ap:.4f}")

    # 4. Tag the model version with the AP and current Date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    client.set_model_version_tag(registry_name, latest_version.version, "prod_eval_ap", str(fresh_ap))
    client.set_model_version_tag(registry_name, latest_version.version, "prod_eval_date", current_date)
    
    logger.info(f"Successfully logged AP {fresh_ap:.4f} and Date {current_date} to {registry_name} v{latest_version.version}")

if __name__ == "__main__":
    evaluate_for_production()