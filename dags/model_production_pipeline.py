import os
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param
from datetime import datetime, timedelta

# Dynamically fetch available models (same logic as training_pipeline)
CONF_MODEL_DIR = "/opt/airflow/conf/model"
try:
    available_models = [f.replace('.yaml', '') for f in os.listdir(CONF_MODEL_DIR) if f.endswith('.yaml')]
except FileNotFoundError:
    available_models = ["cnn3d", "cnn2d", "resnet3d", "spatialcnn"]

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(seconds=10),
}

with DAG(
    dag_id='model_production_promotion',
    default_args=default_args,
    description='Evaluates registered models in parallel and promotes the best to production',
    schedule='@monthly',
    start_date=datetime(2026, 3, 23),
    catchup=False,
    tags=['tornado_capstone', 'production'],
    params={
        "target_year": Param(
            default=2013,
            type="integer",
            description="Year of the Tornet dataset to use for the Thunderdome fair evaluation."
        )
    }
) as dag:

    # The Final Promotion Task
    run_logic_promotion = (
        "cd /opt/airflow && PYTHONPATH=/opt/airflow/src:$PYTHONPATH python src/model_production/promote_to_production.py "
        "tracking.uri='http://mlflow_server:5000' "
    )
    
    promote_best_model = BashOperator(
        task_id='promote_champion_to_production',
        bash_command=run_logic_promotion,
        trigger_rule='all_success' # Ensures all evaluations must finish first
    )

    # Fan-out Evaluation Tasks
    for model_name in available_models:
        run_logic_eval = (
            f"cd /opt/airflow && PYTHONPATH=/opt/airflow/src:$PYTHONPATH python src/model_production/evaluate_for_production.py "
            f"model={model_name} "
            f"api.dataset.target_year={{{{ params.target_year }}}} "
            f"tracking.uri='http://mlflow_server:5000' "
        )

        evaluate_model_task = BashOperator(
            task_id=f'evaluate_prod_{model_name}',
            bash_command=run_logic_eval
        )

        # Map dependencies: Evaluate -> Promote
        evaluate_model_task >> promote_best_model