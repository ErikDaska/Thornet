import dask
import hydra
from omegaconf import DictConfig
import xarray as xr
import torch
import logging
from pathlib import Path

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

VAR_BOUNDS = {
    'DBZ':   {'min': -32.0, 'max': 95.0},
    'VEL':   {'min': -100.0, 'max': 100.0},
    'KDP':   {'min': -10.0, 'max': 10.0},
    'RHOHV': {'min': 0.0, 'max': 1.05},
    'ZDR':   {'min': -8.0, 'max': 15.0},
    'WIDTH': {'min': 0.0, 'max': 30.0}
}


def process_and_save_single_file(file_path: Path, output_dir: Path, scan_idx: int):
    """Processes a single NetCDF file and saves its time steps as PyTorch tensors."""

    # Open a SINGLE file
    ds = xr.open_dataset(file_path, engine="netcdf4")

    # Create MASK and Fill NaNs
    ds['MASK'] = (~ds['DBZ'].isnull()).astype(float)
    fill_values = {'DBZ': -10.0, 'VEL': 0.0, 'KDP': 0.0, 'RHOHV': 0.0, 'ZDR': 0.0, 'WIDTH': 0.0}
    ds = ds.fillna(fill_values)

    # Fast Normalization using pre-defined bounds
    vars_to_scale = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH']
    for var in vars_to_scale:
        xmin = VAR_BOUNDS[var]['min']
        xmax = VAR_BOUNDS[var]['max']
        # Clip values just in case there are outliers beyond our bounds
        ds[var] = ds[var].clip(min=xmin, max=xmax)
        ds[var] = (ds[var] - xmin) / (xmax - xmin)

    # Convert to Tensor
    feature_list = ['DBZ', 'VEL', 'RHOHV', 'ZDR', 'MASK', 'WIDTH']

    # Handle single sweep if it exists
    if 'sweep' in ds.dims:
        ds = ds.isel(sweep=0)

    # Convert to array: (time, channel, azimuth, range)
    data_array = ds[feature_list].to_array(dim='channel')
    data_array = data_array.transpose('time', 'channel', 'azimuth', 'range')

    # Convert directly to tensor
    tensor = torch.from_numpy(data_array.values).float()

    # Save individual time steps
    for i in range(tensor.shape[0]):
        scan = tensor[i]
        out_path = output_dir / f"scan_{scan_idx:04d}.pt"
        torch.save(scan, out_path)
        scan_idx += 1

    ds.close()
    return scan_idx

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def preprocess_data(cfg: DictConfig):
    """
    Retrieves data from DVC (local path), converts it to xarray,
    drops missing values, and converts them into PyTorch tensors.
    """

    logger.info("--- Starting Data Preprocessing Pipeline ---")

    # Access the dataset path using Hydra config
    data_path = Path(cfg.api.dataset.raw_path)

    if not data_path.exists():
        logger.error(f"Data not found at {data_path}. Did you run 'dvc pull'?")
        return

    output_dir = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to xarray
    logger.info(f"Loading NetCDF files from {data_path}...")
    try:
        all_files = sorted(list(data_path.rglob("*.nc")))
        logger.info(f"Found {len(all_files)} files to process.")

        scan_idx = 0
        # Process file-by-file. This takes very little RAM and avoids Dask overhead.
        for i, file_path in enumerate(all_files):
            if i % 10 == 0:
                logger.info(f"Processing file {i}/{len(all_files)}...")

            try:
                scan_idx = process_and_save_single_file(file_path, output_dir, scan_idx)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        logger.info(f"Pipeline completed! Saved {scan_idx} total tensors.")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    preprocess_data()