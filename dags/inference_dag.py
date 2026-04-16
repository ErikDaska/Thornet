"""
inference_dag.py
"""

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner':             'mlops_engineer',
    'depends_on_past':   False,
    'email_on_failure':  False,
    'email_on_retry':    False,
    'retries':           2,
    'retry_delay':       timedelta(minutes=1),
}

with DAG(
    dag_id='tornado_inference_realtime',
    default_args=default_args,
    description='Continuous Inference Pipeline - generates offline_data_fallback.csv every 5 minutes',
    schedule=timedelta(minutes=5),         # Runs every 5 minutes
    start_date=datetime(2026, 4, 10),
    catchup=False,
    max_active_runs=1,             # Prevents concurrent executions
    tags=['tornado_capstone', 'inference', 'realtime'],
) as dag:

    run_inference = BashOperator(
        task_id='run_inference_pipeline',
        bash_command=(
            'cd /opt/airflow && '
            'python src/inference/inference_pipeline.py'
        ),
        env={
            'MLFLOW_TRACKING_URI':    'http://mlflow_server:5000',
            'PREDICTIONS_OUTPUT':     '/opt/airflow/data/offline_data_fallback.csv',
            'PROCESSED_DATA_DIR':     '/opt/airflow/data/processed',
            'RADARS_CSV_PATH':        '/opt/airflow/data/radars/radars.csv',
            'TARGET_YEAR':            '2013',
            'PYTHONPATH':             '/opt/airflow/src',
        },
        append_env=True,
    )