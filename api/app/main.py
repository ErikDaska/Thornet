from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime, time, timezone
import mlflow
import torch
import numpy as np
import pandas as pd
import logging
import os
import glob
import xarray as xr
import re

# Configure isolated logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Thornet API", description="MaaS: Model-as-a-Service for PyTorch inference.")

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MODEL_NAME = "ThornetTornadoPrediction"
ALIAS = "production"
DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/data/processed")
AVAILABLE_DATES = []

VAR_BOUNDS = {
    'DBZ':   {'min': -32.0, 'max': 95.0},
    'VEL':   {'min': -100.0, 'max': 100.0},
    'KDP':   {'min': -10.0, 'max': 10.0},
    'RHOHV': {'min': 0.0, 'max': 1.05},
    'ZDR':   {'min': -8.0, 'max': 15.0},
    'WIDTH': {'min': 0.0, 'max': 30.0}
}

CHANNEL_ORDER = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'MASK']

# Global State for Model
model = None
model_version = "unknown"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def startup_event():
    """Load model and scan inventory on startup."""
    global model, model_version, AVAILABLE_DATES
    
    # 1. Load Model from MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        model_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=ALIAS)
        model_uri = f"models:/{MODEL_NAME}/{model_info.version}"

        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device("cpu"))
        model = model.to(device)
        model.eval()
        model_version = model_info.version
        logger.info(f"Loaded {MODEL_NAME} | Version: {model_version} on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    # 2. Build Inventory
    scan_available_data()

# Load Radar Coordinates Database
try:
    RADAR_DF = pd.read_csv("/data/radars/radars.csv", index_col="radar_id")
    logger.info("Successfully loaded radar coordinates database.")
except Exception as e:
    logger.warning(f"Could not load radars.csv: {e}")
    RADAR_DF = pd.DataFrame()

# --- Schemas ---
class PredictionItem(BaseModel):
    radar_id: str
    sensor: str     # Unified with dashboard/pipeline naming
    lat: float
    lon: float
    tornado_probability: float
    timestamp: str  # Required by Streamlit app

class ForecastResponse(BaseModel):
    target_date: date
    predictions: List[PredictionItem]

class ForecastRequest(BaseModel):
    date_: date

# --- Helpers ---
def load_and_preprocess_file(filepath: str) -> torch.Tensor:
    """
    Reads NetCDF and converts to normalized 5D Tensor [1, 7, 2, 120, 240].
    Dimensions: [Batch, Channels/Variables, Sweeps, Height, Width]
    """
    with xr.open_dataset(filepath, engine="h5netcdf") as ds:
        if 'time' in ds.dims:
            ds = ds.isel(time=0)

        # 1. Fill NaNs and ensure float32
        ds = ds.fillna(0.0)

        # 2. MASK Creation (if missing in the file)
        if 'MASK' not in ds.data_vars:
            if 'DBZ' in ds.data_vars:
                ds['MASK'] = (~ds['DBZ'].isnull()).astype(np.float32)
            else:
                # Use first available var for shape fallback
                first_var = list(ds.data_vars)[0] if ds.data_vars else None
                if first_var:
                    ds['MASK'] = xr.zeros_like(ds[first_var]).astype(np.float32)
                else:
                    # Total fallback
                    ds['MASK'] = xr.DataArray(np.zeros((2, 120, 240), dtype=np.float32), dims=['sweep', 'azimuth', 'range'])

        # 3. Channel Stacking: 7 variables x 2 sweeps = 14 total, but kept 5D
        # Standard TorNet Order: DBZ, VEL, ZDR, RHOHV, KDP, WIDTH, MASK
        n_sweeps = ds.sizes.get('sweep', 1)
        var_channels = []
        
        for var in ['DBZ', 'VEL', 'ZDR', 'RHOHV', 'KDP', 'WIDTH', 'MASK']:
            sweep_data = []
            for s_idx in range(2): 
                # If file only has 1 sweep, repeat it (common for some datasets)
                current_s = s_idx if s_idx < n_sweeps else 0 
                if var in ds.data_vars:
                    # Extract single sweep
                    data = ds[var].isel(sweep=current_s).values.copy()
                    sweep_data.append(data.astype(np.float32))
                else:
                    sweep_data.append(np.zeros((120, 240), dtype=np.float32))
            
            # Stack sweeps for this variable: [2, 120, 240]
            var_channels.append(np.stack(sweep_data, axis=0))

    # Stack all variables: [7, 2, 120, 240]
    stacked_data = np.stack(var_channels, axis=0) 
    return torch.from_numpy(stacked_data).float().unsqueeze(0) # [1, 7, 2, 120, 240]

def adapt_model_input(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Dynamically adapts the standardized 5D TorNet tensor to the model's architecture.
    """
    # 1. Check if model is 3D or 2D by inspecting the first convolutional layer
    is_3d = False
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            is_3d = True
            break
        if isinstance(m, torch.nn.Conv2d):
            is_3d = False
            break

    if not is_3d and x.dim() == 5:
        # Flatten Channels and Sweeps for 2D models: [B, 7, 2, H, W] -> [B, 14, H, W]
        b, c, s, h, w = x.shape
        logger.info(f"Adapting 5D input to 4D for 2D model architecture (Shape: {b}x{c*s}x{h}x{w})")
        return x.view(b, c * s, h, w)
    
    if is_3d and x.dim() == 4:
        # Unlikely but for safety: [B, 14, H, W] -> [B, 7, 2, H, W]
        b, cs, h, w = x.shape
        logger.info(f"Adapting 4D input to 5D for 3D model architecture (Shape: {b}x{cs//2}x2x{h}x{w})")
        return x.view(b, cs // 2, 2, h, w)

    return x

# No main.py, confirma que estas funções estão assim:

def scan_available_data():
    global AVAILABLE_DATES
    # Adicionamos um log para debug
    logger.info(f"Scanning for .nc files in: {DATA_DIR}")
    
    # Procura ficheiros .nc em todas as subpastas
    all_nc_files = glob.glob(os.path.join(DATA_DIR, "**", "*.nc"), recursive=True)
    
    unique_dates = set()
    for f in all_nc_files:
        # Extrai YYMMDD (ex: 131222) do nome do ficheiro
        match = re.search(r"_(\d{6})_", os.path.basename(f))
        if match:
            try:
                # Converte para objeto date e depois para ISO string
                dt = datetime.strptime(match.group(1), "%y%m%d").date()
                unique_dates.add(dt.isoformat())
            except Exception:
                continue
    
    # Atualiza a lista global
    AVAILABLE_DATES = sorted(list(unique_dates), reverse=True)
    logger.info(f"Inventory updated: {len(AVAILABLE_DATES)} dates found.")



# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "FastAPI is running", "model": MODEL_NAME}

@app.get("/health")
def health_check():
    """Detailed health check for dashboard observability."""
    return {
        "status": "online",
        "model": MODEL_NAME,
        "version": model_version,
        "device": str(device),
        "inventory_size": len(AVAILABLE_DATES)
    }

@app.get("/api/v1/inventory")
def get_inventory():
    if not AVAILABLE_DATES:
        scan_available_data()
    return {"dates": AVAILABLE_DATES}


@app.get("/api/v1/radars")
def get_radars():
    if RADAR_DF.empty:
        logger.warning("Data base of radars (RADAR_DF) is empty.")
        return {}
    
    return RADAR_DF.fillna(0.0).to_dict(orient="index")

@app.get("/")
def read_root():
    return {"status": "online", "message": "ThorNet API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/v1/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        date_str = request.date_.strftime("%y%m%d")
        search_pattern = os.path.join(DATA_DIR, "**", f"*_{date_str}_*.nc")
        matched_files = glob.glob(search_pattern, recursive=True)

        results = []
        with torch.no_grad():
            for filepath in matched_files:
                filename = os.path.basename(filepath)
                parts = filename.split("_")
                
                # Extract metadata from filename: processed_PREFIX_DATE_HHMMSS_RADARID_...
                try:
                    # Example parts: ['processed', 'NUL', '130831', '234422', 'KBGM', ...]
                    raw_time = parts[3] # Correct HHMMSS index
                    radar_id = parts[4] # Correct RadarID index
                    
                    # Construct valid ISO timestamp for the Streamlit filter
                    dt_obj = datetime.combine(
                        request.date_, 
                        time(hour=int(raw_time[:2]), minute=int(raw_time[2:4]), second=int(raw_time[4:6]))
                    ).replace(tzinfo=timezone.utc)
                    ts_str = dt_obj.isoformat()
                except:
                    radar_id = "UNKNOWN"
                    ts_str = datetime.combine(request.date_, time(0)).isoformat()

                # Inference
                input_tensor = load_and_preprocess_file(filepath).to(device)
                
                # Dynamic shape adaptation based on model type
                input_tensor = adapt_model_input(model, input_tensor)

                # DIAGNOSTIC: Check input stats
                t_min = input_tensor.min().item()
                t_max = input_tensor.max().item()
                t_mean = input_tensor.mean().item()
                
                raw_output = model(input_tensor)
                prob = torch.sigmoid(raw_output).item()

                logger.info(f"Scan {filename}: Prob={prob:.4f} | Logit={raw_output.item():.4f} | Input[shape={list(input_tensor.shape)}, min={t_min:.4f}, max={t_max:.4f}]")

                # Geo Lookup
                lat, lon = 0.0, 0.0
                if radar_id in RADAR_DF.index:
                    lat = float(RADAR_DF.loc[radar_id, "lat"])
                    lon = float(RADAR_DF.loc[radar_id, "lon"])

                results.append(PredictionItem(
                    radar_id=radar_id,
                    sensor=radar_id, # Mapping radar_id to sensor
                    lat=lat,
                    lon=lon,
                    tornado_probability=prob,
                    timestamp=ts_str
                ))

        return ForecastResponse(target_date=request.date_, predictions=results)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")