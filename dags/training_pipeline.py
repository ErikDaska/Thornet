from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param
from datetime import datetime, timedelta
import os

CONF_MODEL_DIR = "/opt/airflow/conf/model"
try:
    available_models = [f.replace('.yaml', '') for f in os.listdir(CONF_MODEL_DIR) if f.endswith('.yaml')]
except FileNotFoundError:
    # Fallback safety
    available_models = ["cnn3d", "cnn2d", "resnet3d", "spatialcnn"]

dropdown_options = ["all"] + available_models

# Default arguments applied to all tasks in the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(seconds=10),
}

# DAG context
with DAG(
    dag_id='training_evaluating_model',
    default_args=default_args,
    description='Training Pipeline for Tornado Forecasting',
    schedule='@monthly',
    start_date=datetime(2026, 3, 23),
    catchup=False,
    tags=['tornado_capstone', 'training'],
    params={
        "target_model": Param(
            default="all",
            enum=dropdown_options,
            description="Select 'all' to run every model, or pick a specific one to train."
        ),
        "target_year": Param(
            default=2013,
            type="integer",
            description="Year of the Tornet dataset"
        )
    }
) as dag:

    for model_name in available_models:

        # BROKEN OUT FOR READABILITY: Notice the addition of the target_year parameter override
        run_logic_training = (
            f"if [ '{{{{ params.target_model }}}}' = 'all' ] || [ '{{{{ params.target_model }}}}' = '{model_name}' ]; then "
            f"cd /opt/airflow && PYTHONPATH=/opt/airflow/src:$PYTHONPATH python src/training/train_model.py "
            f"model={model_name} "
            f"api.dataset.target_year={{{{ params.target_year }}}} "
            f"tracking.uri='http://mlflow_server:5000' "
            f"tracking.experiment_name='Airflow_Automated_Run'; "
            "else "
            f"echo 'Skipping {model_name} training based on UI selection.'; "
            "exit 99; "
            "fi"
        )

        # Model Training
        train_model = BashOperator(
            task_id=f'train_{model_name}_model',
            bash_command=run_logic_training
        )

        # BROKEN OUT FOR READABILITY: Notice the addition of the target_year parameter override
        run_logic_evaluation = (
            f"if [ '{{{{ params.target_model }}}}' = 'all' ] || [ '{{{{ params.target_model }}}}' = '{model_name}' ]; then "
            f"cd /opt/airflow && PYTHONPATH=/opt/airflow/src:$PYTHONPATH python src/evaluation/evaluate_model.py "
            f"model={model_name} "
            f"api.dataset.target_year={{{{ params.target_year }}}} "
            f"tracking.uri='http://mlflow_server:5000' "
            f"tracking.experiment_name='Airflow_Automated_Run'; "
            "else "
            f"echo 'Skipping {model_name} evaluation based on UI selection.'; "
            "exit 99; "
            "fi"
        )

        # Model Evaluation
        evaluate_model = BashOperator(
            task_id=f'evaluate_{model_name}_model',
            bash_command=run_logic_evaluation
        )

        train_model >> evaluate_model