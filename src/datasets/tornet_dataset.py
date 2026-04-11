from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import torch

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TornetDataset(Dataset):
    def __init__(self, data_dir: Path, catalog_path: Path, variables=None):
        self.files = sorted(list(data_dir.rglob("*.nc")))

        if variables is None:
            self.variables = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'MASK']
        else:
            self.variables = variables

        if not self.files:
            logger.warning(f"No .nc files found in {data_dir}!")

        # --- 1. Load Labels from Catalog ---
        logger.info(f"Loading labels from catalog: {catalog_path}")
        if catalog_path.exists():
            self.catalog = pd.read_csv(catalog_path)
        else:
            logger.error(f"Catalog not found at {catalog_path}. Labels will default to 0.0!")
            self.catalog = pd.DataFrame()

        # --- 2. PRE-COMPUTE LABELS (Vectorized - Lightning Fast) ---
        self.label_map = {}
        logger.info("Pre-computing label mapping...")
        if not self.catalog.empty:
            # Filter the catalog ONCE to get only tornadoes
            tor_catalog = self.catalog[self.catalog['category'] == 'TOR']

            # Combine all columns into a single giant string block for instant searching
            tor_text_blob = tor_catalog.to_string()

            for f in self.files:
                original_filename = f.name.replace("processed_", "")
                # Instant check: Is this filename anywhere inside the Tornado text block?
                if original_filename in tor_text_blob:
                    self.label_map[str(f)] = 1.0
                else:
                    self.label_map[str(f)] = 0.0

        # --- 3. Build an Index Map (Skipping xarray open!) ---
        self.index_map = []
        logger.info("Building dataset index...")
        for f in self.files:
            # We assume 1 time step per file (idx 0) to bypass slow NetCDF reading
            self.index_map.append((f, 0))

        logger.info(f"Dataset mapped: {len(self.index_map)} total 3D scans available.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, time_idx = self.index_map[idx]

        # We only open the file right when we need it for the batch
        with xr.open_dataset(file_path, engine="netcdf4") as ds:
            if 'time' in ds.dims:
                step_ds = ds.isel(time=time_idx)
            else:
                step_ds = ds

            channels = []
            for var in self.variables:
                if var in step_ds.data_vars:
                    data = step_ds[var].values
                else:
                    shape = (step_ds.sizes.get('sweep', 1),
                             step_ds.sizes.get('azimuth', 1),
                             step_ds.sizes.get('range', 1))
                    data = np.zeros(shape, dtype=np.float32)

                channels.append(data)

            features = np.stack(channels, axis=0)
            features_tensor = torch.from_numpy(features).float()

        label = self.label_map.get(str(file_path), 0.0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return features_tensor, label_tensor
