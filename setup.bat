@echo off
echo ===================================================
echo  Starting TorNet MLOps Environment Setup (Windows)
echo ===================================================

echo [1/6] Verifying directory structure...
if not exist "..\emi_remote" mkdir "..\emi_remote"
if not exist "data\raw" mkdir "data\raw"
if not exist "src" mkdir "src"

echo [2/6] Spinning up MLflow Tracking Server...
docker compose up -d
echo Waiting 15s for MLflow to initialize...
timeout /t 15 /nobreak >nul

echo [3/6] Configuring Python environment...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
:: Added netcdf4 and h5netcdf to ensure xarray works
pip install -r requirements.txt zenodo-get netcdf4 h5netcdf -q

echo [4/6] Checking dataset status (TorNet 2013)...
if not exist "data\raw\tornet_2013" (
    echo    Dataset not found locally. Initiating download...
    mkdir "data\raw\tornet_2013"
    zenodo_get 12636522 -o data\raw\
    echo    Extracting archive to data\raw\tornet_2013...
    tar -xzf data\raw\tornet_2013.tar.gz -C data\raw\tornet_2013\
    del data\raw\tornet_2013.tar.gz
) else (
    echo    Dataset 2013 found locally.
)

echo [5/6] Syncing with local DVC remote...
dvc init --no-scm -f >nul 2>&1
dvc remote list | findstr "localremote" >nul 2>&1
if errorlevel 1 (
    dvc remote add -d localremote ..\emi_remote
)

:: Track only this specific year. Scale by adding dvc add for 2014, etc later.
dvc add data\raw\tornet_2013
dvc push

echo [6/6] Executing Data Ingestion Pipeline...
:: Pass the directory as an argument to make the python script flexible
python src\data_ingestion.py

echo ===================================================
echo  Setup Complete! Check MLflow at http://127.0.0.1:5000
echo ===================================================
pause