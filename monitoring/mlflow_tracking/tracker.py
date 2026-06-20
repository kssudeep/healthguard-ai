"""monitoring/mlflow_tracking/tracker.py"""
from __future__ import annotations
import logging
import mlflow
from config.settings import settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("healthguard-ai")

    def start_run(self, run_name: str, tags: dict = None) -> str:
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            return run.info.run_id

    def log_metrics(self, run_id: str, metrics: dict):
        try:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"MLflow metric log failed: {e}")

    def log_param(self, run_id: str, key: str, value):
        try:
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_param(key, str(value))
        except Exception as e:
            logger.warning(f"MLflow param log failed: {e}")

    def end_run(self, run_id: str):
        try:
            mlflow.end_run()
        except Exception:
            pass

    def get_run(self, run_id: str) -> dict:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return {
            "run_id": run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "status": run.info.status,
        }
