from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.sdk import Param


# Default arguments applied to all tasks in the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG context
with DAG(
    dag_id='training_evaluating_model',
    default_args=default_args,
    description='End-to-End MLOps Pipeline for Tornado Forecasting',
    schedule='@daily',
    start_date=datetime(2026, 3, 23),
    catchup=False,
    tags=['tornado_capstone', 'training']
) as dag:

    # Model Training
    train_model = BashOperator(
        task_id='train_thornet_cnn_model',
        bash_command='cd /opt/airflow && python src/training/train_model.py tracking.experiment_name="Airflow_Automated_Run" tracking.uri="http://mlflow_server:5000"'
    )

    # Model Evaluation
    evaluate_model = BashOperator(
        task_id='evaluate_best_model',
        bash_command='cd /opt/airflow && python src/evaluation/evaluate_model.py'
    )

    train_model >> evaluate_model