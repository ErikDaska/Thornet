from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from typing import List
from datetime import date
import mlflow
import torch
import numpy as np
import pandas as pd
import logging
import os
import glob
import xarray as xr
import re

# Configure isolated logging for the API microservice
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Thornet API", description="MaaS: Model-as-a-Service for PyTorch inference.")

# State Initialization: Dynamically load the Production Model and its metadata
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "Tornet-2DCNN"
ALIAS = "Production"
DATA_DIR = "/data/raw/tornet_2014"


VAR_BOUNDS = {
    'DBZ':   {'min': -32.0, 'max': 95.0},
    'VEL':   {'min': -100.0, 'max': 100.0},
    'KDP':   {'min': -10.0, 'max': 10.0},
    'RHOHV': {'min': 0.0, 'max': 1.05},
    'ZDR':   {'min': -8.0, 'max': 15.0},
    'WIDTH': {'min': 0.0, 'max': 30.0}
}

# Matches your TornetDataset exactly
CHANNEL_ORDER = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'MASK']


# Load Model
try:
 mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 client = MlflowClient()
 model_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=ALIAS)
 MODEL_URI = f"models:/{MODEL_NAME}/{model_info.version}"

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = mlflow.pytorch.load_model(MODEL_URI).to(device)
 model.eval()
 logger.info(f"Loaded {MODEL_NAME} | Version: {model_info.version} on {device}")
except Exception as e:
 logger.error(f"Failed to load model from MLflow Registry: {e}")
 model = None

# Load Radar Coordinates
try:
    # Assuming radars.csv is inside your /data volume
    RADAR_DF = pd.read_csv("/data/radars.csv", index_col="radar_id")
    logger.info("Successfully loaded radar coordinates database.")
except Exception as e:
    logger.warning(f"Could not load radars.csv. Coordinates will be missing. Error: {e}")
    RADAR_DF = pd.DataFrame()

# Pydantic Schemas
class ForecastRequest(BaseModel):
    date_: date = Field(..., description="The date to forecast a Tornado (YYYY-MM-DD)")

class PredictionItem(BaseModel):
 radar_id: str
 lat: float
 lon: float
 tornado_probability: float

class ForecastResponse(BaseModel):
 target_date: date
 predictions: List[PredictionItem]

# --- Helper Functions ---
def load_and_preprocess_file(filepath: str) -> torch.Tensor:
 """
 Reads a raw NetCDF file, strips the time dimension, applies MASK and normalization,
 and converts it to a PyTorch tensor with a batch dimension.
 """
 # 1. Open raw file
 ds = xr.open_dataset(filepath, engine="netcdf4")

 # 2. Handle the time dimension just like your training dataset
 if 'time' in ds.dims:
  ds = ds.isel(time=0)  # Take the first (and likely only) time step

 # 3. Create MASK and Fill NaNs
 if 'DBZ' in ds.data_vars:
  ds['MASK'] = (~ds['DBZ'].isnull()).astype(float)
 else:
  # Fallback to zeros matching the shape of the 3D volume
  ds['MASK'] = xr.zeros_like(ds[list(ds.data_vars)[0]])

 ds = ds.fillna(0.0)

 # 4. Normalize Variables
 for var, bounds in VAR_BOUNDS.items():
  if var in ds.data_vars:
   min_val = bounds['min']
   max_val = bounds['max']
   ds[var] = (ds[var] - min_val) / (max_val - min_val)
   ds[var] = ds[var].clip(0.0, 1.0)

 # 5. Extract and stack data
 channels_data = []
 for var in CHANNEL_ORDER:
  if var in ds.data_vars:
   arr = ds[var].values
   channels_data.append(arr)
  else:
   # Replicating your dataset's zero-padding for missing variables
   shape = (
    ds.sizes.get('sweep', 1),
    ds.sizes.get('azimuth', 1),
    ds.sizes.get('range', 1)
   )
   channels_data.append(np.zeros(shape, dtype=np.float32))

 ds.close()

 # 6. Stack into (C, Sweeps, Azimuth, Range)
 stacked_data = np.stack(channels_data, axis=0)

 # 7. Convert to PyTorch Tensor and add Batch Dimension -> (1, C, S, A, R)
 tensor_data = torch.tensor(stacked_data, dtype=torch.float32).unsqueeze(0)

 return tensor_data


# --- Endpoints ---
@app.post("/api/v1/forecast", response_model=ForecastResponse)  # Fixed response model
def generate_forecast(request: ForecastRequest):
 if not model:
  raise HTTPException(status_code=503, detail="Inference model is currently unavailable.")

 try:
  # Format date to TorNet's YYMMDD format (e.g., 2014-01-11 -> '140111')
  date_yy_mm_dd = request.date_.strftime("%y%m%d")
  logger.info(f"Scanning for date identifier: {date_yy_mm_dd}")

  # Search both train and test directories for files matching this date
  # Pattern matches: PREFIX_YYMMDD_TIME_RADAR_...nc
  search_pattern = os.path.join(DATA_DIR, "**", f"*_{date_yy_mm_dd}_*.nc")
  matched_files = glob.glob(search_pattern, recursive=True)

  if not matched_files:
   logger.warning(f"No radar data found for date {request.date_}")
   return ForecastResponse(target_date=request.date_, predictions=[])

  results = []

  # Process each found file
  with torch.no_grad():
   for filepath in matched_files:
    filename = os.path.basename(filepath)

    # Extract Radar ID using Regex (Assuming format: NUL_140111_105240_KBMX_...)
    # It splits by underscore and grabs the 4th element (index 3)
    try:
     radar_id = filename.split("_")[3]
    except IndexError:
     logger.warning(f"Could not parse radar ID from filename: {filename}")
     continue

    # Preprocess Data
    input_tensor = load_and_preprocess_file(filepath).to(device)

    # Run Inference
    raw_output = model(input_tensor)
    probability = torch.sigmoid(raw_output).item()  # .item() extracts single float

    # Lookup Coordinates
    lat, lon = 0.0, 0.0
    if not RADAR_DF.empty and radar_id in RADAR_DF.index:
     lat = float(RADAR_DF.loc[radar_id, "lat"])
     lon = float(RADAR_DF.loc[radar_id, "lon"])

    # Append to results
    results.append(
     PredictionItem(
      radar_id=radar_id,
      lat=lat,
      lon=lon,
      tornado_probability=probability
     )
    )

  return ForecastResponse(
   target_date=request.date_,
   predictions=results
  )

 except Exception as e:
  logger.error(f"Inference failed: {e}", exc_info=True)
  raise HTTPException(status_code=500, detail="Internal server error during inference processing.")