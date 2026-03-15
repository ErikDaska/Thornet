import argparse
import logging
from pathlib import Path

import dvc.api
import mlflow
import xarray as xr

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 2. HELPER FUNCTIONS ---
def get_dvc_lineage(data_path: Path) -> str:
    """Retrieves the DVC hash for the dataset to ensure strict data lineage."""
    logger.info(f"Extracting DVC lineage for: {data_path}")
    try:
        resource_url = dvc.api.get_url(path=str(data_path), repo=".")
        # Extract the hash from the DVC path string
        return Path(resource_url).name
    except Exception as e:
        logger.warning(f"Failed to retrieve DVC hash (Is the folder tracked by DVC?): {e}")
        return "unknown_hash"

def get_directory_size_mb(data_path: Path) -> float:
    """Calculates the total physical size of the dataset directory in Megabytes."""
    total_size = sum(
        f.stat().st_size for f in data_path.rglob('*') 
        if f.is_file() and not f.is_symlink()
    )
    return total_size / (1024 * 1024)

def extract_metadata(data_path: Path) -> dict:
    """Scans the dataset to extract structural and scientific metadata."""
    logger.info(f"Scanning dataset structure in {data_path}...")
    
    nc_files = list(data_path.rglob("*.nc"))
    
    metadata = {
        "total_files": len(nc_files),
        "total_size_mb": get_directory_size_mb(data_path),
        "radar_variables": "None",
        "spatial_dimensions": "None"
    }
    
    if nc_files:
        try:
            # Explicitly set engine to resolve Xarray backend errors
            # Use 'with' to ensure the file is properly closed after reading
            with xr.open_dataset(nc_files[0], engine="netcdf4") as ds:
                metadata["radar_variables"] = ", ".join(list(ds.data_vars.keys()))
                metadata["spatial_dimensions"] = str(dict(ds.sizes))
        except Exception as e:
            logger.warning(f"Could not read .nc file structure. Check dependencies: {e}")
            
    return metadata

# --- 3. MAIN PIPELINE ---
def main():
    # Setup Argument Parser for dynamic execution
    parser = argparse.ArgumentParser(description="TorNet Data Ingestion Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/raw/tornet_2013", help="Target dataset directory")
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000", help="MLflow Tracking URI")
    parser.add_argument("--experiment", type=str, default="TorNet_Data_Ingestion", help="MLflow Experiment Name")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    
    # Dynamic Run Name based on the specific folder (e.g., "tornet_2013")
    run_name = f"ingestion_{data_path.name}"

    logger.info(f"--- Starting Data Ingestion Pipeline for: {data_path.name} ---")
    
    if not data_path.exists():
        logger.error(f"Dataset directory not found: {data_path}. Please check your download step.")
        return

    # Extract Data Lineage and Metadata
    dvc_hash = get_dvc_lineage(data_path)
    dataset_metadata = extract_metadata(data_path)

    # Configure MLflow
    logger.info(f"Connecting to MLflow Tracking Server at {args.mlflow_uri}...")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    # Execute MLflow Run
    logger.info(f"Initiating MLflow run: '{run_name}'")
    with mlflow.start_run(run_name=run_name):
        
        # Log Parameters (Metadata/Config)
        mlflow.log_params({
            "dataset_name": "MIT-LL TorNet",
            "dataset_source_path": str(data_path),
            "dvc_data_hash": dvc_hash,
            "radar_variables": dataset_metadata["radar_variables"],
            "spatial_dimensions": dataset_metadata["spatial_dimensions"]
        })
        
        # Log Metrics (Numeric values)
        mlflow.log_metrics({
            "total_files_ingested": dataset_metadata["total_files"],
            "dataset_size_mb": round(dataset_metadata["total_size_mb"], 2)
        })
        
        logger.info("Successfully logged all data lineage and metadata to MLflow!")

if __name__ == "__main__":
    main()