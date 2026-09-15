"""monitoring/mlflow_tracking/tracker.py"""
import logging
import os
import mlflow

logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "healthguard-ai"

mlflow.set_tracking_uri(MLFLOW_URI)
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    logger.warning(f"MLflow experiment setup failed: {e}")


class MLflowTracker:
    def __init__(self):
        self._runs = {}

    def start_run(self, run_name: str, tags: dict = None) -> str:
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags or {})
            run_id = run.info.run_id
            self._runs[run_id] = run
            return run_id
        except Exception as e:
            logger.warning(f"MLflow start_run failed: {e}")
            return "local-run"

    def log_metrics(self, run_id: str, metrics: dict):
        try:
            mlflow.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"MLflow log_metrics failed: {e}")

    def log_param(self, run_id: str, key: str, value):
        try:
            mlflow.log_param(key, str(value)[:250])
        except Exception as e:
            logger.warning(f"MLflow log_param failed: {e}")

    def end_run(self, run_id: str):
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow end_run failed: {e}")

    def get_run(self, run_id: str) -> dict:
        try:
            run = mlflow.get_run(run_id)
            return {
                "run_id": run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "status": run.info.status,
            }
        except Exception as e:
            return {"run_id": run_id, "metrics": {}, "params": {}, "status": "FINISHED"}
