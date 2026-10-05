import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Add parent directory to sys.path to allow importing src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model


default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_pipeline_dag",
    default_args=default_args,
    description="Automated ML pipeline for testing, training, validation, and deployment",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "ci-cd", "classification"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data_task",
        python_callable=preprocess_data,
        op_kwargs={"data_dir": "data/processed"},
    )

    train_task = PythonOperator(
        task_id="train_model_task",
        python_callable=train_model,
        op_kwargs={
            "train_path": "data/processed/train.csv",
            "model_output_path": "models/model.joblib",
            "metadata_output_path": "models/training_metadata.json",
        },
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model_task",
        python_callable=evaluate_model,
        op_kwargs={
            "test_path": "data/processed/test.csv",
            "model_path": "models/model.joblib",
            "metrics_path": "models/metrics.json",
            "accuracy_threshold": 0.85,
        },
    )

    deploy_task = PythonOperator(
        task_id="deploy_model_task",
        python_callable=deploy_model,
        op_kwargs={
            "source_model_dir": "models",
            "target_deploy_dir": "production_model",
            "metrics_path": "models/metrics.json",
        },
    )

    # Define DAG task execution workflow
    preprocess_task >> train_task >> evaluate_task >> deploy_task
