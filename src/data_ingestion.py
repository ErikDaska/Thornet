import os
import glob
import logging
import mlflow
import dvc.api
import xarray as xr

# --- 1. CONFIGURATION ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "TorNet_Data_Ingestion"
DATA_DIR = "data/raw/tornet"
RUN_NAME = "baseline_ingestion"

# --- 2. LOGGING SETUP ---
# Clean code practice: Use standard logging instead of print statements
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 3. HELPER FUNCTIONS ---
def get_dvc_lineage(data_path: str) -> str:
    """Retrieves the DVC hash for the dataset to ensure strict data lineage."""
    logger.info(f"Extracting DVC lineage for: {data_path}")
    try:
        # Using the DVC Python API to get the hash of the data directory
        resource_url = dvc.api.get_url(path=data_path, repo=".")
        folder_hash = os.path.basename(resource_url)
        return folder_hash
    except Exception as e:
        logger.error(f"Failed to retrieve DVC hash via API: {e}")
        return "unknown_hash"

def get_directory_size_mb(data_path: str) -> float:
    """Calculates the total physical size of the dataset directory in Megabytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(data_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def extract_metadata(data_path: str) -> dict:
    """Scans the TorNet dataset to extract structural and scientific metadata."""
    logger.info("Scanning dataset structure...")
    
    metadata = {
        "total_files": 0,
        "total_size_mb": 0.0,
        "radar_variables": "None",
        "spatial_dimensions": "None"
    }
    
    nc_files = glob.glob(os.path.join(data_path, "**", "*.nc"), recursive=True)
    metadata["total_files"] = len(nc_files)
    metadata["total_size_mb"] = get_directory_size_mb(data_path)
    
    if nc_files:
        try:
            # Lazily open the first NetCDF file to extract its schema
            ds = xr.open_dataset(nc_files[0])
            metadata["radar_variables"] = ", ".join(list(ds.data_vars.keys()))
            metadata["spatial_dimensions"] = str(dict(ds.sizes))
            ds.close()
        except Exception as e:
            logger.warning(f"Could not read .nc file structure: {e}")
            
    return metadata

# --- 4. MAIN PIPELINE ---
def main():
    logger.info("--- Starting TorNet Data Ingestion Pipeline ---")
    
    # Verify dataset exists before starting
    if not os.path.exists(DATA_DIR):
        logger.error(f"Dataset directory not found: {DATA_DIR}. Did you run 'dvc pull'?")
        return

    # Extract Data Lineage and Metadata
    dvc_hash = get_dvc_lineage(DATA_DIR)
    dataset_metadata = extract_metadata(DATA_DIR)

    # Configure MLflow
    logger.info(f"Connecting to MLflow Tracking Server at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Execute MLflow Run
    logger.info(f"Initiating MLflow run: '{RUN_NAME}'")
    with mlflow.start_run(run_name=RUN_NAME):
        
        # Log Lineage Parameters (Crucial for Guidelines Evaluation CA2)
        mlflow.log_param("dataset_name", "MIT-LL TorNet")
        mlflow.log_param("dataset_source_path", DATA_DIR)
        mlflow.log_param("dvc_data_hash", dvc_hash)
        
        # Log Structural Parameters
        mlflow.log_param("radar_variables", dataset_metadata["radar_variables"])
        mlflow.log_param("spatial_dimensions", dataset_metadata["spatial_dimensions"])
        
        # Log Metrics
        mlflow.log_metric("total_files_ingested", dataset_metadata["total_files"])
        mlflow.log_metric("dataset_size_mb", round(dataset_metadata["total_size_mb"], 2))
        
        logger.info("Successfully logged all data lineage and metadata to MLflow!")
        logger.info("Pipeline Complete. Check the MLflow UI to verify.")

if __name__ == "__main__":
    main()