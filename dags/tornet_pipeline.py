from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
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
    dag_id='thornet_mlops_data_pipeline',
    default_args=default_args,
    description='End-to-End MLOps Pipeline for Tornado Forecasting',
    schedule='@weekly',
    start_date=datetime(2026, 3, 23),
    catchup=False,
    max_active_runs=1,
    render_template_as_native_obj=True,
    tags=['tornado_capstone', 'training'],
    params={
        "target_year": Param(2013, type="integer", description="Year of the Tornet dataset to process (2013-2022)")
    }
) as dag:
    # Task 1: Data Ingestion
    ingest_data = BashOperator(
        task_id='ingest_tornet_data',
        bash_command=(
            'cd /opt/airflow && python src/data_ingestion/data_ingestion.py '
            'tracking.uri="http://mlflow_server:5000" '
            'tracking.experiment_name="Airflow_Automated_Run" '
            'api.dataset.target_year="{{ params.target_year }}"'
        )
    )
    # Task 2: Data Processing
    data_process = BashOperator(
        task_id='process_tornet_data',
        bash_command=(
            'cd /opt/airflow && python src/data_processing/new_data_processing.py '
            'tracking.uri="http://mlflow_server:5000" '
            'tracking.experiment_name="Airflow_Automated_Run" '
            'api.dataset.target_year="{{ params.target_year }}"'
        )
    )

    # Trigger the Training DAG
    trigger_training = TriggerDagRunOperator(
        task_id='trigger_training_dag',
        trigger_dag_id='training_evaluating_model',
        conf={"target_year": "{{ params.target_year }}"},
        wait_for_completion=False
    )

    ingest_data >> data_process >> trigger_training