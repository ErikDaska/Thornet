"""
inference_pipeline.py

Continuous Inference Script for the Tornado Alert Dashboard.

"""

import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CONFIGURATION
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "Airflow_Automated_Run")
OUTPUT_CSV          = Path(os.getenv("PREDICTIONS_OUTPUT", "/opt/airflow/data/dados_para_teste.csv"))
PROCESSED_DATA_DIR  = Path(os.getenv("PROCESSED_DATA_DIR", "/opt/airflow/data/processed"))
TARGET_YEAR         = int(os.getenv("TARGET_YEAR", "2013"))
MAX_SAMPLES         = int(os.getenv("MAX_INFERENCE_SAMPLES", "50"))  # Limited for demo purposes
ALERT_THRESHOLD_KM  = float(os.getenv("ALERT_THRESHOLD_KM", "200.0"))




def _run_model_inference() -> pd.DataFrame | None:
    """
    Attempts to load the model from MLflow and run inference on processed data.
    Returns a DataFrame with predictions, or None on failure.
    """
    try:
        import mlflow
        import mlflow.pytorch
        from torch.utils.data import DataLoader, Subset
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from datasets.tornet_dataset import TornetDataset

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri("sqlite:////mlflow/mlflow.db")

        logger.info(f"🔍 Searching for the latest model in '{EXPERIMENT_NAME}'...")
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string="tags.mlflow.runName LIKE '3dcnn_training_%'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if runs.empty:
            logger.warning("No training runs found in MLflow.")
            return None

        run_id   = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"
        logger.info(f"✅ Model found: run_id={run_id}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        logger.info(f"📦 Model loaded on {device}.")

        # Look for processed data
        data_dir = PROCESSED_DATA_DIR / str(TARGET_YEAR)
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return None

        # Catalog for labels
        catalog_candidates = [
            PROCESSED_DATA_DIR.parent / "raw" / f"tornet_{TARGET_YEAR}" / "catalog.csv",
            PROCESSED_DATA_DIR.parent / "raw" / "catalog.csv",
        ]
        catalog_path = next((p for p in catalog_candidates if p.exists()), None)

        if catalog_path is None:
            logger.warning("Catalog not found. Using zero-labels (no ground truth).")
            # Create a temporary empty catalog
            catalog_path = PROCESSED_DATA_DIR / "_empty_catalog.csv"
            pd.DataFrame(columns=["filename", "category"]).to_csv(catalog_path, index=False)

        dataset = TornetDataset(data_dir=data_dir, catalog_path=catalog_path)

        if len(dataset) == 0:
            logger.warning("Empty dataset — no .nc files found.")
            return None

        # Select random subset for the demo
        n_samples = min(MAX_SAMPLES, len(dataset))
        indices   = random.sample(range(len(dataset)), n_samples)
        subset    = Subset(dataset, indices)
        loader    = DataLoader(subset, batch_size=16, shuffle=False, num_workers=0)

        records = []
        now     = datetime.now(timezone.utc)

        with torch.no_grad():
            sample_idx = 0
            for inputs, _ in loader:
                inputs  = inputs.to(device)
                outputs = model(inputs)
                probs   = torch.sigmoid(outputs).cpu().numpy().flatten()

                for prob in probs:
                    detected = int(prob > 0.5)
                    # Note: TorNet scans don't have native GPS. Using 0,0 since simulation was removed.
                    lat, lon = 0.0, 0.0 
                    records.append({
                        "timestamp":        now.isoformat(),
                        "scan_id":          f"scan_{indices[sample_idx]:06d}",
                        "tornado_detected": detected,
                        "probability":      round(float(prob), 4),
                        "latitude":         lat,
                        "longitude":        lon,
                        "alert_level":      _get_alert_level(float(prob), detected),
                        "source":           "model_inference",
                    })
                    sample_idx += 1

        logger.info(f"✅ Inference complete: {len(records)} scans processed.")
        return pd.DataFrame(records)

    except Exception as e:
        logger.error(f"Inference failed with real model: {e}", exc_info=True)
        return None


def main():
    logger.info("=" * 60)
    logger.info("🌪️  TorNet — Continuous Inference Pipeline")
    logger.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Try real inference only
    df = _run_model_inference()

    if df is not None and not df.empty:
        # Save to CSV only if we have real predictions
        df.to_csv(OUTPUT_CSV, index=False)
        n_tornadoes = int(df["tornado_detected"].sum())
        logger.info(f"💾 Real predictions saved to '{OUTPUT_CSV}'")
        logger.info(f"📊 Total scans: {len(df)} | Tornadoes detected: {n_tornadoes}")
    else:
        logger.error("🔴 No real model predictions available. Skipping CSV update.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()