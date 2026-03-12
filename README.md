# MLOps Pipeline: TorNet Radar Data Ingestion

This repository contains the foundational MLOps infrastructure for processing the MIT-LL TorNet dataset. 

The current pipeline demonstrates strict separation of code and data, data versioning using DVC with a simulated local remote, and structural metadata logging via MLflow.

## 📂 Project Structure (Submission State)
Because the TorNet dataset exceeds academic submission file size limits, the raw data and the DVC remote are **not** included in this ZIP file. 

Upon extraction, your directory should look like this:
```text
📦 emi_project            # This repository
 ┣ 📂 data
 ┃ ┗ 📂 raw               # Empty folder (Data goes here)
 ┣ 📂 src
 ┃ ┗ 📜 data_ingestion.py # Baseline pipeline script
 ┣ 📜 .dvcignore
 ┣ 📜 data/raw/tornet.dvc # The DVC tracking file holding the data lineage hash
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

---

## ⚙️ Environment Setup

### 1. Configure the Python Virtual Environment

Navigate into the `emi_project` directory and set up an isolated Python environment.

**For Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🗄️ Data Download & DVC Configuration

Because the dataset is not included, you must download it and initialize the simulated local remote manually to reproduce the environment.

### Step 1: Download the Data

1. Download the TorNet dataset subset used for this project from the official source (https://github.com/mit-ll/tornet?tab=readme-ov-file).
2. Extract the downloaded NetCDF (`.nc`) files directly into the following folder inside this project:
`emi_project/data/raw/tornet/`

### Step 2: Recreate the Simulated Remote

Create an empty folder one level above the project directory to act as the simulated remote storage.
**Linux / macOS:** `mkdir ../emi_remote`
**Windows:** `mkdir ..\emi_remote`

### Step 3: Configure DVC and Push Data

Now, link DVC to the newly created remote and push the heavy data to it. Ensure your terminal is inside the `emi_project` folder.

```bash
# 1. Tell DVC where the simulated remote is located
dvc remote add -d mylocalremote ../emi_remote

# 2. Tell DVC to re-calculate the hashes just in case OS differences altered them
dvc add data/raw/tornet

# 3. Push the heavy files into the remote cache
dvc push
```

*The MLOps data infrastructure is now fully prepared and ready for ingestion!*

---

## 🚀 Running the MLflow Tracking Server

To ensure data lineage and pipeline execution are logged correctly, you must start the MLflow server in a dedicated terminal window before running the pipeline.

*(Make sure your virtual environment is activated in this terminal)*

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Keep this terminal open and running in the background. You can view the dashboard at [http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000).

---

## 🏃‍♂️ Executing the Ingestion Pipeline

With the MLflow server running, open a **second terminal window**, activate your virtual environment, and execute the data ingestion script:

```
python src\data_ingestion.py
```

### Expected Output

The script will interface with the DVC API to extract the secure data hash, lazily load the first NetCDF file using `xarray` to extract scientific radar variables (e.g., DBZ, VEL), and log all structural metadata and the DVC hash directly to the `TorNet_Data_Ingestion` experiment in MLflow.



