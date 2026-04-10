import hydra
from omegaconf import DictConfig
import xarray as xr
import logging
from pathlib import Path

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- NORMALIZATION BOUNDS ---
VAR_BOUNDS = {
    'DBZ':   {'min': -32.0, 'max': 95.0},
    'VEL':   {'min': -100.0, 'max': 100.0},
    'KDP':   {'min': -10.0, 'max': 10.0},
    'RHOHV': {'min': 0.0, 'max': 1.05},
    'ZDR':   {'min': -8.0, 'max': 15.0},
    'WIDTH': {'min': 0.0, 'max': 30.0}
}

def process_and_save_single_file(file_path: Path, output_dir: Path):
    """Processes a raw NetCDF file, normalizes it, and saves it as a compressed NetCDF."""
    
    # 1. Open raw file (keeps the 3D structure intact!)
    ds = xr.open_dataset(file_path, engine="netcdf4")

    # 2. Create MASK and Fill NaNs
    # MASK tracks where the radar actually returned a signal vs empty sky
    if 'DBZ' in ds.data_vars:
        ds['MASK'] = (~ds['DBZ'].isnull()).astype(float)
    
    ds = ds.fillna(0.0)

    # 3. Normalize Variables
    for var, bounds in VAR_BOUNDS.items():
        if var in ds.data_vars:
            min_val = bounds['min']
            max_val = bounds['max']
            # Min-max scaling to [0, 1]
            ds[var] = (ds[var] - min_val) / (max_val - min_val)
            # Clip to ensure no outliers break the [0, 1] range
            ds[var] = ds[var].clip(0.0, 1.0)

    # 4. Save as a compressed NetCDF
    output_file = output_dir / f"processed_{file_path.name}"
    
    # Apply zlib compression to every data variable to save massive amounts of disk space
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
    
    ds.to_netcdf(output_file, engine="netcdf4", encoding=encoding)
    ds.close()


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def process_data(cfg: DictConfig):
    logger.info("--- Starting Data Preprocessing Pipeline (Compressed 3D) ---")

    data_path = Path(cfg.api.dataset.raw_path)

    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Did you run 'dvc pull'?")
        return

    output_dir = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(list(data_path.rglob("*.nc")))
    logger.info(f"Found {len(all_files)} files to process.")

    for i, file_path in enumerate(all_files):
        if i % 10 == 0:
            logger.info(f"Processing file {i}/{len(all_files)}: {file_path.name}")
        
        try:
            process_and_save_single_file(file_path, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    logger.info("Data preprocessing complete!")

if __name__ == "__main__":
    process_data()