from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param
from datetime import datetime, timedelta

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
    dag_id='model_production_promotion',
    default_args=default_args,
    description='Evaluates all registered models on the test set and promotes the best to production',
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

    # Run logic specifically for triggering the production script with hydra overrides
    run_logic_production = (
        "cd /opt/airflow && PYTHONPATH=/opt/airflow/src:$PYTHONPATH python src/production/model_production.py "
        "api.dataset.target_year={{ params.target_year }} "
        "tracking.uri='http://mlflow_server:5000' "
    )

    # Model Production Task
    promote_best_model = BashOperator(
        task_id='evaluate_and_promote_champion',
        bash_command=run_logic_production
    )

    promote_best_model