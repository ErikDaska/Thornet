import logging
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score

import hydra
from omegaconf import DictConfig
import mlflow
from mlflow.tracking import MlflowClient
from datasets.tornet_dataset import TornetDataset
from training.models import get_model # ensures models are in context

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def thunderdome(cfg: DictConfig):
    logger.info("🌩️ Welcome to the Thunderdome! 🌩️")
    
    mlflow.set_tracking_uri(cfg.tracking.uri)
    client = MlflowClient()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Dataset (Once for all models)
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv"
    test_indices_path = Path(cfg.paths.processed_data_dir) / f"test_indices_{cfg.api.dataset.target_year}.csv"

    logger.info("Loading latest test dataset for fair comparison...")
    full_dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)
    test_indices = pd.read_csv(test_indices_path)['test_index'].tolist()
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 2. Find all registered models
    registered_models = client.search_registered_models(filter_string="name ILIKE 'Tornet-%'")
    if not registered_models:
        logger.error("No registered models found in MLflow. Skipping Thunderdome.")
        return

    best_global_ap = 0.0
    best_model_name = None
    best_model_version = None

    # 3. The Gauntlet (Sequential testing to avoid OOM)
    for rm in registered_models:
        try:
            # Get latest version of this architecture
            latest_version = client.get_latest_versions(name=rm.name)[0]
            model_uri = f"models:/{rm.name}/{latest_version.version}"
            logger.info(f"--> Evaluating {rm.name} (Version {latest_version.version})")

            # Load Model
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

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            
            # Calculate fresh metric on new data
            fresh_ap = average_precision_score(y_true, y_scores)
            logger.info(f"    Fresh AP Score: {fresh_ap:.4f}")

            # Keep track of the global best
            if fresh_ap > best_global_ap:
                best_global_ap = fresh_ap
                best_model_name = rm.name
                best_model_version = latest_version.version

        except Exception as e:
            logger.error(f"Failed to evaluate {rm.name}: {e}")

        finally:
            # CRITICAL: Purge model from VRAM so the next one can load
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # 4. Crown the Global Champion
    if best_model_name:
        logger.info(f"🏆 GLOBAL CHAMPION: {best_model_name} (Version {best_model_version}) with AP: {best_global_ap:.4f}")
        
        # Apply the @production alias to the absolute winner
        client.set_registered_model_alias(
            name=best_model_name,
            alias="production",
            version=best_model_version
        )
        logger.info("The @production alias has been updated! Downstream APIs will now serve this model.")

if __name__ == "__main__":
    thunderdome()