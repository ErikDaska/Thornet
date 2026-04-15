import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import re

import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import sys
from datasets.tornet_dataset import TornetDataset

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CONFIGURATION
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
OUTPUT_CSV          = Path(os.getenv("PREDICTIONS_OUTPUT", "/opt/airflow/data/offline_data_fallback.csv"))
PROCESSED_DATA_DIR  = Path(os.getenv("PROCESSED_DATA_DIR", "/opt/airflow/data/processed"))
RADARS_CSV_PATH     = Path(os.getenv("RADARS_CSV_PATH", "/opt/airflow/data/radars/radars.csv"))
TARGET_YEAR         = int(os.getenv("TARGET_YEAR", "2013"))


def _load_radar_lookup(radars_path: Path) -> pd.DataFrame:
    """Loads the NEXRAD radar station database for coordinate lookups."""
    if not radars_path.exists():
        logger.warning(f"Radar database not found at {radars_path}.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(radars_path, index_col="radar_id")
        logger.info(f"Loaded {len(df)} radar stations from {radars_path}.")
        return df
    except Exception as e:
        logger.error(f"Failed to load radars.csv: {e}")
        return pd.DataFrame()


def _extract_radar_id(filename: str) -> str | None:
    """
    Extracts the radar station ID from a TorNet filename.
    Expected format: PREFIX_YYMMDD_TIME_RADARID_...nc
    E.g.: processed_NUL_140111_105240_KBMX_..._V06.nc -> KBMX
    """
    try:
        clean_name = filename.replace("processed_", "")
        parts = clean_name.split("_")
        if len(parts) >= 4:
            radar_id = parts[3]
            if len(radar_id) == 4 and radar_id[0] in ("K", "P", "T"):
                return radar_id
    except Exception:
        pass
    return None


def _run_model_inference() -> pd.DataFrame | None:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        model_name = "ThornetTornadoPrediction"
        alias = "production"
        model_uri = f"models:/{model_name}@{alias}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}.")

        data_dir = PROCESSED_DATA_DIR / str(TARGET_YEAR)
        catalog_path = PROCESSED_DATA_DIR.parent / "raw" / f"tornet_{TARGET_YEAR}" / "catalog.csv"
        if not catalog_path.exists():
            logger.warning(f"Catalog not found at {catalog_path}. Creating a dummy catalog.")
            catalog_path = PROCESSED_DATA_DIR / "_empty_catalog.csv"
            pd.DataFrame(columns=["filename", "category"]).to_csv(catalog_path, index=False)
        dataset = TornetDataset(data_dir=data_dir, catalog_path=catalog_path)
        radar_db = _load_radar_lookup(RADARS_CSV_PATH)

        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        records = []
        now = datetime.now(timezone.utc)
        global_idx = 0
        skipped = 0

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                for prob in probs:
                    file_path = dataset.index_map[global_idx][0]
                    filename = file_path.name
                    
                    radar_id = _extract_radar_id(filename)

                    match = re.search(r"_(\d{6})_(\d{6})_", filename)
                    if match:
                        d, t = match.groups()
                        actual_ts = f"20{d[:2]}-{d[2:4]}-{d[4:]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"
                    else:
                        actual_ts = now.isoformat()

                    if radar_id and not radar_db.empty and radar_id in radar_db.index:
                        lat = float(radar_db.loc[radar_id, "lat"])
                        lon = float(radar_db.loc[radar_id, "lon"])
                        
                        records.append({
                            "timestamp": actual_ts,
                            "scan_id": f"scan_{global_idx:06d}",
                            "tornado_detected": int(prob > 0.5),
                            "probability": round(float(prob), 4),
                            "latitude": lat,
                            "longitude": lon,
                            "sensor": radar_id,
                            "source": "thornet_production_v2",
                        })
                    else:
                        skipped += 1
                    
                    global_idx += 1

        logger.info(f"Inference complete: {len(records)} records generated, {skipped} skipped.")
        return pd.DataFrame(records)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return None

def main():
    logger.info("=" * 60)
    logger.info("🌪️  TorNet — Production Inference Pipeline v2")
    logger.info("=" * 60)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = _run_model_inference()

    if df is not None and not df.empty:
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Results saved to {OUTPUT_CSV}")
    else:
        logger.error("Inference produced no results.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()