"""
MLflow Tracking Service - Manages experiment tracking with local file storage.

All MLflow operations are wrapped in error handling so that tracking failures
never break the core model training pipeline.
"""
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

_mlflow = None


def _get_mlflow():
    """Lazy load MLflow to avoid import errors if not installed."""
    global _mlflow
    if _mlflow is None:
        import mlflow
        _mlflow = mlflow
    return _mlflow


class MLflowTracker:
    """
    Session-aware MLflow experiment tracker.

    Uses local file-based tracking (./mlruns/ directory).
    Each session gets its own MLflow experiment.

    All public methods catch exceptions internally so callers
    never need error handling.
    """

    def __init__(self, session_id: Optional[str] = None):
        self._session_id = session_id
        self._experiment_id: Optional[str] = None
        self._enabled: bool = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize MLflow with local file tracking."""
        try:
            mlflow = _get_mlflow()

            tracking_uri = f"file://{os.path.abspath('./mlruns')}"
            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = f"spark_tune_{self._session_id or 'default'}"
            experiment = mlflow.get_experiment_by_name(experiment_name)

            if experiment is None:
                self._experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self._experiment_id = experiment.experiment_id

            self._enabled = True
            logger.info(f"MLflow initialized: experiment={experiment_name}, id={self._experiment_id}")

        except ImportError:
            logger.warning("MLflow not installed -- tracking disabled")
            self._enabled = False
        except Exception as e:
            logger.warning(f"MLflow initialization failed -- tracking disabled: {e}")
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def experiment_id(self) -> Optional[str]:
        return self._experiment_id

    def log_training_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        training_time: Optional[float] = None,
    ) -> Optional[str]:
        """
        Log a complete training run to MLflow.

        Returns MLflow run_id string if successful, None otherwise.
        """
        if not self._enabled:
            return None

        try:
            mlflow = _get_mlflow()

            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=run_name,
            ) as run:
                for key, value in params.items():
                    mlflow.log_param(key, value)

                for key, value in metrics.items():
                    if value is not None:
                        mlflow.log_metric(key, float(value))

                if training_time is not None:
                    mlflow.log_metric("training_time_seconds", training_time)

                mlflow.set_tag("session_id", self._session_id or "default")
                mlflow.set_tag("platform", "spark_tune")
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, str(value))

                run_id = run.info.run_id
                logger.info(f"MLflow run logged: {run_name} (run_id={run_id})")
                return run_id

        except Exception as e:
            logger.warning(f"Failed to log MLflow run '{run_name}': {e}")
            return None

    def log_xgboost_run(
        self,
        params: Dict[str, Any],
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        training_time: Optional[float] = None,
    ) -> Optional[str]:
        """Log an XGBoost training run with prefixed train/test metrics."""
        combined_metrics = {}
        for key, value in train_metrics.items():
            if value is not None:
                combined_metrics[f"train_{key}"] = value
        for key, value in test_metrics.items():
            if value is not None:
                combined_metrics[f"test_{key}"] = value

        return self.log_training_run(
            run_name="XGBoost Training",
            params=params,
            metrics=combined_metrics,
            tags={"model_type": "xgboost", "framework": "spark_xgboost"},
            training_time=training_time,
        )

    def log_baseline_run(
        self,
        model_name: str,
        metrics: Dict[str, float],
        training_time: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Log a baseline model run."""
        return self.log_training_run(
            run_name=f"Baseline - {model_name}",
            params=params or {"model_name": model_name},
            metrics=metrics,
            tags={"model_type": "baseline", "model_name": model_name},
            training_time=training_time,
        )

    def log_automl_run(
        self,
        config: Dict[str, Any],
        best_score: float,
        search_time: float,
        problem_type: str,
        metric: str,
    ) -> Optional[str]:
        """Log an AutoML run."""
        return self.log_training_run(
            run_name="AutoML - LightAutoML",
            params=config,
            metrics={"best_score": best_score, "search_time": search_time},
            tags={
                "model_type": "automl",
                "framework": "lightautoml",
                "problem_type": problem_type,
                "metric": metric,
            },
            training_time=search_time,
        )

    def log_feature_comparison_run(
        self,
        model_name: str,
        feature_set: str,
        metrics: Dict[str, float],
        training_time: float,
        num_features: int,
    ) -> Optional[str]:
        """Log a feature comparison model run."""
        return self.log_training_run(
            run_name=f"{model_name} - {feature_set} features",
            params={
                "model_name": model_name,
                "feature_set": feature_set,
                "num_features": num_features,
            },
            metrics=metrics,
            tags={
                "model_type": "feature_comparison",
                "model_name": model_name,
                "feature_set": feature_set,
            },
            training_time=training_time,
        )

    def get_experiment_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current experiment."""
        if not self._enabled or not self._experiment_id:
            return None

        try:
            mlflow = _get_mlflow()
            experiment = mlflow.get_experiment(self._experiment_id)
            return {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "lifecycle_stage": experiment.lifecycle_stage,
            }
        except Exception as e:
            logger.warning(f"Failed to get experiment info: {e}")
            return None

    def get_runs(
        self,
        max_results: int = 50,
        order_by: str = "start_time DESC",
    ) -> List[Dict[str, Any]]:
        """
        Get runs for the current experiment.

        Returns list of dicts with: run_id, run_name, status,
        start_time, params, metrics, tags.
        """
        if not self._enabled or not self._experiment_id:
            return []

        try:
            mlflow = _get_mlflow()
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                max_results=max_results,
                order_by=[order_by],
            )

            results = []
            for _, row in runs.iterrows():
                run_data: Dict[str, Any] = {
                    "run_id": row.get("run_id"),
                    "run_name": row.get("tags.mlflow.runName", ""),
                    "status": row.get("status", ""),
                    "start_time": str(row.get("start_time", "")),
                    "end_time": str(row.get("end_time", "")),
                    "params": {},
                    "metrics": {},
                    "tags": {},
                }

                for col in runs.columns:
                    if col.startswith("params."):
                        param_name = col[len("params."):]
                        run_data["params"][param_name] = row[col]
                    elif col.startswith("metrics."):
                        metric_name = col[len("metrics."):]
                        value = row[col]
                        if value is not None and str(value) != "nan":
                            run_data["metrics"][metric_name] = float(value)
                    elif col.startswith("tags.") and not col.startswith("tags.mlflow."):
                        tag_name = col[len("tags."):]
                        if row[col] is not None and str(row[col]) != "nan":
                            run_data["tags"][tag_name] = str(row[col])

                results.append(run_data)

            return results

        except Exception as e:
            logger.warning(f"Failed to get MLflow runs: {e}")
            return []


_session_trackers: Dict[str, MLflowTracker] = {}


def get_mlflow_tracker(session_id: Optional[str] = None) -> MLflowTracker:
    """Get or create an MLflowTracker for a session."""
    key = session_id or "default"
    if key not in _session_trackers:
        _session_trackers[key] = MLflowTracker(session_id=key)
    return _session_trackers[key]
