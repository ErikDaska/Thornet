"""
inference_pipeline.py
---------------------
Continuous Inference Script for the Tornado Alert Dashboard.

Executed periodically by Airflow (inference_dag.py).
Logic:
  1. Connects to MLflow and loads the most recent trained model.
  2. Iterates over samples from the test dataset (processed .nc files).
  3. Runs inference using the Tornet2DCNN.
  4. Assigns realistic simulated US geographic coordinates to each scan.
  5. Saves results to /opt/airflow/data/latest_predictions.csv

FALLBACK MODE: If no model or processed data is available,
the script generates 100% synthetic predictions to ensure the demo works.
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
OUTPUT_CSV          = Path(os.getenv("PREDICTIONS_OUTPUT", "/opt/airflow/data/latest_predictions.csv"))
PROCESSED_DATA_DIR  = Path(os.getenv("PROCESSED_DATA_DIR", "/opt/airflow/data/processed"))
TARGET_YEAR         = int(os.getenv("TARGET_YEAR", "2013"))
MAX_SAMPLES         = int(os.getenv("MAX_INFERENCE_SAMPLES", "50"))  # Limited for demo purposes
ALERT_THRESHOLD_KM  = float(os.getenv("ALERT_THRESHOLD_KM", "200.0"))

# Approximate bounding box for Continental US (where TorNet data was collected)
US_LAT_MIN, US_LAT_MAX = 25.0, 50.0
US_LON_MIN, US_LON_MAX = -125.0, -67.0

# High tornado activity zones ("Tornado Alley") for realistic positive predictions
TORNADO_ALLEY_REGIONS = [
    # (lat_center, lon_center, lat_spread, lon_spread, name)
    (37.0, -97.5, 3.0, 3.0, "Kansas"),
    (35.5, -97.5, 3.0, 3.0, "Oklahoma"),
    (32.0, -96.5, 3.0, 4.0, "North Texas"),
    (38.5, -94.0, 2.0, 2.0, "Missouri"),
    (34.5, -89.5, 2.5, 2.5, "Mississippi"),
    (36.0, -86.5, 2.0, 2.0, "Tennessee"),
    (34.0, -85.5, 2.0, 2.0, "Alabama"),
    (43.0, -96.0, 2.0, 2.0, "South Dakota"),
    (41.0, -95.5, 2.0, 2.0, "Nebraska"),
    (35.0, -100.0, 3.0, 4.0, "West Texas"),
]


def _random_us_coords() -> tuple[float, float]:
    """Generates random coordinates within the continental US."""
    lat = random.uniform(US_LAT_MIN, US_LAT_MAX)
    lon = random.uniform(US_LON_MIN, US_LON_MAX)
    return round(lat, 4), round(lon, 4)


def _random_tornado_coords() -> tuple[float, float]:
    """
    Generates coordinates within a high-activity tornado region.
    Makes the demo geographically more realistic.
    """
    region = random.choice(TORNADO_ALLEY_REGIONS)
    lat_c, lon_c, lat_s, lon_s, _ = region
    lat = random.gauss(lat_c, lat_s * 0.4)
    lon = random.gauss(lon_c, lon_s * 0.4)
    lat = max(US_LAT_MIN, min(US_LAT_MAX, lat))
    lon = max(US_LON_MIN, min(US_LON_MAX, lon))
    return round(lat, 4), round(lon, 4)


def _get_alert_level(probability: float, detected: int) -> str:
    """Classifies the alert level based on probability."""
    if detected == 0:
        return "NONE"
    if probability >= 0.85:
        return "CRITICAL"
    if probability >= 0.65:
        return "HIGH"
    return "MODERATE"


def _generate_synthetic_predictions(n: int = 40) -> pd.DataFrame:
    """
    FALLBACK: Generates N realistic synthetic predictions for the demo
    when no model or data is available.
    Tornado proportion is ~20% (reflects actual TorNet dataset).
    """
    logger.warning("⚠️  SYNTHETIC FALLBACK MODE: Generating simulated predictions.")
    records = []
    now = datetime.now(timezone.utc)

    for i in range(n):
        prob = random.betavariate(1.5, 6.0)  # Skewed towards 0 (mostly non-tornado)
        detected = 1 if prob > 0.5 else 0

        if detected == 1:
            lat, lon = _random_tornado_coords()
        else:
            lat, lon = _random_us_coords()

        records.append({
            "timestamp":        now.isoformat(),
            "scan_id":          f"synthetic_{i:04d}",
            "tornado_detected": detected,
            "probability":      round(prob, 4),
            "latitude":         lat,
            "longitude":        lon,
            "alert_level":      _get_alert_level(prob, detected),
            "source":           "synthetic_fallback",
        })

    return pd.DataFrame(records)


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
                    lat, lon = _random_tornado_coords() if detected else _random_us_coords()
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

    # Try real inference; fallback to synthetic if it fails
    df = _run_model_inference()

    if df is None or df.empty:
        df = _generate_synthetic_predictions(n=MAX_SAMPLES)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    n_tornadoes = int(df["tornado_detected"].sum())
    logger.info(f"💾 CSV saved to '{OUTPUT_CSV}'")
    logger.info(f"📊 Total scans: {len(df)} | Tornadoes detected: {n_tornadoes}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()