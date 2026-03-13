# MLOps Pipeline: TorNet Radar Data Ingestion

This repository contains the foundational MLOps infrastructure for processing the MIT-LL TorNet dataset.

The current pipeline demonstrates strict separation of code and data, data versioning using DVC with a simulated local remote, structural metadata logging via MLflow, and infrastructure automation using Docker Compose.

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
 ┣ 📜 docker-compose.yml  # Infrastructure configuration for MLflow
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┗ 📜 setup.sh            # Automated environment and pipeline execution script

```

---

## 🗄️ Step 1: Data Download (Action Required)

Because the dataset is not included in the submission, you must download the raw data before running the automated setup. The `setup.sh` script is designed to abort safely if the data is missing to protect the DVC lineage.

1. Download a subset of the TorNet dataset (e.g., the 2013 split) from the github (https://github.com/mit-ll/tornet) and place them directly inside the `data/raw/tornet/` directory.*

---

## 🚀 Step 2: Automated Environment & Pipeline Execution

Once the raw data is placed in the correct folder, the entire MLOps infrastructure and data ingestion pipeline can be launched with a single command.

Ensure **Docker Desktop** is running, then execute the setup script from the root of the `emi_project` directory:

**For Linux / macOS:**

```bash
chmod +x setup.sh
./setup.sh

```

**For Windows (Git Bash or WSL):**

```bash
bash setup.sh

```

### ⚙️ What the Setup Script Does:

To ensure flawless reproducibility, the `setup.sh` script automatically performs the following sequence:

1. **Safety Check:** Verifies the dataset exists locally so DVC hashes are not accidentally overwritten.
2. **Infrastructure Initialization:** Uses `docker-compose up -d` to spin up a persistent MLflow tracking server on port `5000`.
3. **Environment Configuration:** Creates an isolated Python virtual environment (`venv`) and installs all necessary dependencies from `requirements.txt`.
4. **Data Versioning:** Configures a simulated local DVC remote (`../emi_remote`), calculates the local data hashes to account for OS differences, and pushes the heavy `.nc` files to the local cache.
5. **Pipeline Execution:** Runs `src/data_ingestion.py` to extract scientific radar metadata via `xarray` and log the strict DVC data lineage directly to MLflow.

---

## 📊 Viewing the Results

Once the script completes successfully, you can verify the experiment tracking and data lineage by accessing the MLflow UI.

Open your web browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Under the `TorNet_Data_Ingestion` experiment, you will find the logged run containing the exact DVC hash, total dataset size, file counts, and multidimensional radar variable configurations.