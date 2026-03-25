import argparse
import logging
from pathlib import Path
import subprocess

import dvc.api
import mlflow
import xarray as xr
import hydra
from omegaconf import OmegaConf, DictConfig

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS ---
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
            with xr.open_dataset(nc_files[0], engine="netcdf4") as ds:
                metadata["radar_variables"] = ", ".join(list(ds.data_vars.keys()))
                metadata["spatial_dimensions"] = str(dict(ds.sizes))
        except Exception as e:
            logger.warning(f"Could not read .nc file structure. Check dependencies: {e}")
            
    return metadata

def run_command(command, working_dir=None):
    """Utility function to run shell commands and capture output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=working_dir,
            capture_output=True, 
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}\nError: {e.stderr}")
        return None

def get_dvc_lineage(path):
    """Retrieves the DVC hash for the dataset to ensure strict data lineage."""
    res = run_command(f"dvc list . {path} --dvc-only")
    return res.strip() if res else "unknown_hash"

# --- MAIN PIPELINE ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    year = cfg.api.dataset.target_year
    dir_name = cfg.api.dataset.dir_name
    run_name = f"ingestion_{dir_name}"

    logger.info(f"--- Starting {cfg.project_name} Data Ingestion for: {year} ---")

    raw_base_dir = Path("data/raw")
    data_path = raw_base_dir / f"tornet_{year}" 
    # Check if data already exists locally (DVC-tracked). If not, download from Zenodo and add to DVC.
    if not data_path.exists():
        logger.warning(f"Dataset directory not found: {data_path}. Getting data from API")
        raw_base_dir.mkdir(parents=True, exist_ok=True)
        # Get the Zenodo ID for the specified year from the config mapping
        zenodo_id = cfg.api.dataset.zenodo_mapping.get(year)
        if not zenodo_id:
            logger.error(f"No Zenodo ID found for year {year}")
            return


        logger.info(f"Downloading year {year} (ID: {zenodo_id})...")
        try: # Try-except to catch any issues with downloading or extracting the data

            run_command(f"python -m zenodo_get {zenodo_id} -o {raw_base_dir}")
            
  
            archive_path = raw_base_dir / f"tornet_{year}.tar.gz"
            # Check if the archive was downloaded successfully before attempting to extract
            if archive_path.exists():

                data_path.mkdir(parents=True, exist_ok=True)
                run_command(f"tar -xzf {archive_path} -C {data_path} --strip-components=1")
                
                archive_path.unlink() 
                logger.info(f"Data for year {year} extracted successfully.")
            else:
                logger.error(f"Archive not found: {archive_path}")
                return
        except Exception as e:
            logger.error(f"Failed: {e}")
            return
        
        # Add the new data directory to DVC tracking
        logger.info(f"Adding to DVC...")
        try: 
            run_command(f"dvc add {data_path}")
            run_command(f"git add {data_path}.dvc")
            run_command(f"dvc push")
        except Exception as e:
            logger.error(f"DVC error: {e}")
            return

    # Extract Data Lineage and Metadata
    dvc_hash = get_dvc_lineage(data_path)
    dataset_metadata = extract_metadata(data_path)

    # Configure MLflow
    logger.info(f"Connecting to MLflow Tracking Server at {cfg.tracking.uri}...")

    mlflow.set_tracking_uri(cfg.tracking.uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)

    # Execute MLflow Run
    logger.info(f"Initiating MLflow run: '{run_name}'")
    with mlflow.start_run(run_name=run_name):
        # Log Parameters
        mlflow.log_params({
            "project": cfg.project_name,
            "dataset_year": year,
            "dataset_source_path": str(data_path),
            "dvc_data_hash": dvc_hash,
            "radar_variables": dataset_metadata.get("radar_variables", "None"),
            "spatial_dimensions": dataset_metadata.get("spatial_dimensions", "None")
        })

        # Log Metrics
        mlflow.log_metrics({
            "total_files_ingested": dataset_metadata["total_files"],
            "dataset_size_mb": round(dataset_metadata["total_size_mb"], 2)
        })

        logger.info("Successfully logged all data lineage and metadata to MLflow!")

if __name__ == "__main__":
    main()