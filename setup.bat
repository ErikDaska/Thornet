@echo off
echo =========================================
echo  Starting TorNet MLOps Environment Setup
echo =========================================

echo [1/6] Verifying directory structure...
if not exist "..\emi_remote" mkdir "..\emi_remote"
if not exist "data\raw" mkdir "data\raw"
if not exist "src" mkdir "src"
if not exist "config" mkdir "config"

echo [2/6] Spinning up MLflow Tracking Server...
docker compose up -d
echo Waiting 15s for MLflow to initialize...
timeout /t 15 /nobreak >nul

echo [3/6] Configuring Python environment...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt zenodo-get netcdf4 h5netcdf -q

echo [3.5/6] Reading configuration from Hydra YAML...
:: Pass any command line arguments (%*) to the bridge script
python src\get_config.py %*

:: Execute the temp file to load the variables, then clean it up
call temp_env.bat
del temp_env.bat

SET DATA_DIR=tornet_%DATA_YEAR%

echo Target Year: %DATA_YEAR%
echo Zenodo ID: %ZENODO_ID%

echo [4/6] Checking dataset status (TorNet %DATA_YEAR%)...
if not exist "data\raw\%DATA_DIR%" (
    echo    Dataset not found locally. Initiating download...
    mkdir "data\raw\%DATA_DIR%"

    :: We use standard percent signs here now!
    zenodo_get %ZENODO_ID% -o data\raw\

    echo    Extracting archive to data\raw\%DATA_DIR%...
    tar -xzf data\raw\%DATA_DIR%.tar.gz -C data\raw\%DATA_DIR%\
    del data\raw\%DATA_DIR%.tar.gz
) else (
    echo    Dataset %DATA_YEAR% found locally.
)

echo [5/6] Syncing with local DVC remote...
dvc init --no-scm -f >nul 2>&1
dvc remote list | findstr "localremote" >nul 2>&1
if errorlevel 1 (
    dvc remote add -d localremote ..\emi_remote
)

:: Track only the dynamically loaded year
dvc add data\raw\%DATA_DIR%
dvc push

echo [6/6] Executing Data Ingestion Pipeline...
:: Hydra will automatically pick up config/config.yaml
python src\data_ingestion\data_ingestion.py tracking.uri="http://127.0.0.1:5000" %*

echo ===================================================
echo  Setup Complete! Check MLflow at http://127.0.0.1:5000
echo ===================================================
pause