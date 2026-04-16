import logging
from datetime import datetime
import mlflow
import hydra
from omegaconf import DictConfig
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def promote_to_production(cfg: DictConfig):
    logger.info("Starting Final Model Promotion Phase...")
    
    mlflow.set_tracking_uri(cfg.tracking.uri)
    client = MlflowClient(tracking_uri=cfg.tracking.uri)
    unified_model_name = "ThornetTornadoPrediction"
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 1. Get current production AP
    current_prod_ap = 0.0
    try:
        prod_version_data = client.get_model_version_by_alias(unified_model_name, "production")
        current_prod_ap = float(prod_version_data.tags.get("Model_AP", 0.0))
        logger.info(f"Current Production Model AP: {current_prod_ap:.4f}")
    except Exception:
        logger.info("No current production model found. Baseline AP set to 0.0.")

    # 2. Find all recently evaluated registered models
    registered_models = client.search_registered_models(filter_string="name ILIKE 'ThorNet-%'")
    
    best_global_ap = 0.0
    best_model_name = None
    best_model_version = None
    best_run_id = None

    for rm in registered_models:
        try:
            latest_version = client.get_latest_versions(name=rm.name)[0]
            tags = latest_version.tags
            
            eval_date = tags.get("prod_eval_date")
            eval_ap_str = tags.get("prod_eval_ap")

            # Only consider models evaluated TODAY to prevent using stale metrics
            if eval_date == current_date and eval_ap_str is not None:
                eval_ap = float(eval_ap_str)
                logger.info(f"Candidate: {rm.name} v{latest_version.version} | AP: {eval_ap:.4f}")
                
                if eval_ap > best_global_ap:
                    best_global_ap = eval_ap
                    best_model_name = rm.name
                    best_model_version = latest_version.version
                    best_run_id = latest_version.run_id
            else:
                logger.info(f"Skipping {rm.name} v{latest_version.version} (No evaluation from today).")
                
        except Exception as e:
            logger.warning(f"Could not parse candidate {rm.name}: {e}")

    # 3. Crown the Global Champion
    if best_model_name and best_global_ap > current_prod_ap:
        logger.info(f"👑 GLOBAL CHAMPION: {best_model_name} v{best_model_version} with AP: {best_global_ap:.4f}")
        
        model_uri = f"runs:/{best_run_id}/model"
        
        # Register to the unified endpoint
        new_version = mlflow.register_model(
            model_uri=model_uri,
            name=unified_model_name,
            tags={
                "Winning_Architecture": best_model_name,
                "Original_Version": best_model_version,
                "Model_AP": f"{best_global_ap:.4f}",
                "Promotion_Date": current_date
            }
        )

        # Archive older versions
        try:
            latest_versions = client.search_model_versions(f"name='{unified_model_name}'")
            for v in latest_versions:
                if v.version != new_version.version:
                    client.transition_model_version_stage(
                        name=unified_model_name,
                        version=v.version,
                        stage="Archived"
                    )
        except Exception as e:
            logger.warning(f"Could not archive old versions: {e}")
        
        # Assign @production alias
        client.set_registered_model_alias(
            name=unified_model_name,
            alias="production",
            version=new_version.version
        )
        
        logger.info(f"Success! {unified_model_name} v{new_version.version} promoted to @production.")
    else:
        logger.info(f"No models surpassed the current production AP of {current_prod_ap:.4f}. Production remains unchanged.")

if __name__ == "__main__":
    promote_to_production()