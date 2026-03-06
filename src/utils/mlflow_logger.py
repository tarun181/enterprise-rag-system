import mlflow
import yaml
import os


def log_experiment_run(config_dirs: list, eval_metrics: dict, run_name: str = "qwen_hybrid_rag"):
    """Logs configs, pipeline params, and evaluation metrics to MLflow."""
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("Enterprise_Knowledge_Assistant")

    with mlflow.start_run(run_name=run_name):
        # 1. Log configs
        for config_path in config_dirs:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                # Flatten dict for MLflow logging
                flat_config = {f"{k1}_{k2}": v2 for k1, v1 in config_data.items() for k2, v2 in v1.items()}
                mlflow.log_params(flat_config)

        # 2. Log Evaluation Metrics
        mlflow.log_metrics(eval_metrics)

        # 3. Log Prompt Template Artifacts (example)
        mlflow.log_artifact("configs/agent.yaml")

        print(f"Successfully logged run '{run_name}' to MLflow.")