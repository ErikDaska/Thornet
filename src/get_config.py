import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Get the year and look up the ID
    year = cfg.api.dataset.target_year
    zenodo_id = cfg.api.dataset.zenodo_mapping[year]

    # Write them to a temporary batch file for Windows to read
    with open("temp_env.bat", "w") as f:
        f.write(f"SET DATA_YEAR={year}\n")
        f.write(f"SET ZENODO_ID={zenodo_id}\n")

if __name__ == "__main__":
    main()