"""
Session state data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class SessionState:
    """
    Serializable session state.

    Contains all metadata needed to restore a session, with file paths
    pointing to large objects stored on filesystem.
    """

    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    # Pipeline state flags
    has_data: bool = False
    has_problem: bool = False
    has_features: bool = False
    has_preprocessed: bool = False
    has_model: bool = False

    # Problem definition (JSON-serializable)
    problem: Optional[Dict[str, Any]] = None

    # Schema info (summarized for storage)
    schema_summary: Optional[Dict[str, Any]] = None

    # Metrics (JSON-serializable)
    metrics: Optional[Dict[str, Any]] = None

    # Data profile report
    profile_report: Optional[Dict[str, Any]] = None

    # File paths (stored in Redis, files on filesystem)
    data_file_path: Optional[str] = None
    features_file_path: Optional[str] = None
    transformed_file_path: Optional[str] = None
    model_path: Optional[str] = None

    # Feature engineering metadata
    feature_names: Optional[List[str]] = None
    feature_output_col: Optional[str] = None
    feature_idx_name_mapping: Optional[Dict[str, str]] = None

    # Dataset metadata
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: Optional[List[str]] = None

    # Model comparison data
    model_comparison: Optional[Dict[str, Any]] = None

    def to_redis_dict(self) -> Dict[str, str]:
        """Convert to Redis hash-compatible dict (all values as strings)."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "pipeline_state": json.dumps(
                {
                    "has_data": self.has_data,
                    "has_problem": self.has_problem,
                    "has_features": self.has_features,
                    "has_preprocessed": self.has_preprocessed,
                    "has_model": self.has_model,
                }
            ),
            "problem": json.dumps(self.problem) if self.problem else "",
            "schema_summary": json.dumps(self.schema_summary)
            if self.schema_summary
            else "",
            "metrics": json.dumps(self.metrics) if self.metrics else "",
            "profile_report": json.dumps(self.profile_report)
            if self.profile_report
            else "",
            "data_file_path": self.data_file_path or "",
            "features_file_path": self.features_file_path or "",
            "transformed_file_path": self.transformed_file_path or "",
            "model_path": self.model_path or "",
            "feature_names": json.dumps(self.feature_names)
            if self.feature_names
            else "",
            "feature_output_col": self.feature_output_col or "",
            "feature_idx_name_mapping": json.dumps(self.feature_idx_name_mapping)
            if self.feature_idx_name_mapping
            else "",
            "row_count": str(self.row_count) if self.row_count is not None else "",
            "column_count": str(self.column_count)
            if self.column_count is not None
            else "",
            "columns": json.dumps(self.columns) if self.columns else "",
            "model_comparison": json.dumps(self.model_comparison)
            if self.model_comparison
            else "",
        }

    @classmethod
    def from_redis_dict(cls, data: Dict[str, str]) -> "SessionState":
        """Reconstruct SessionState from Redis hash."""
        pipeline_state = json.loads(data.get("pipeline_state", "{}"))

        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            has_data=pipeline_state.get("has_data", False),
            has_problem=pipeline_state.get("has_problem", False),
            has_features=pipeline_state.get("has_features", False),
            has_preprocessed=pipeline_state.get("has_preprocessed", False),
            has_model=pipeline_state.get("has_model", False),
            problem=json.loads(data["problem"]) if data.get("problem") else None,
            schema_summary=json.loads(data["schema_summary"])
            if data.get("schema_summary")
            else None,
            metrics=json.loads(data["metrics"]) if data.get("metrics") else None,
            profile_report=json.loads(data["profile_report"])
            if data.get("profile_report")
            else None,
            data_file_path=data.get("data_file_path") or None,
            features_file_path=data.get("features_file_path") or None,
            transformed_file_path=data.get("transformed_file_path") or None,
            model_path=data.get("model_path") or None,
            feature_names=json.loads(data["feature_names"])
            if data.get("feature_names")
            else None,
            feature_output_col=data.get("feature_output_col") or None,
            feature_idx_name_mapping=json.loads(data["feature_idx_name_mapping"])
            if data.get("feature_idx_name_mapping")
            else None,
            row_count=int(data["row_count"]) if data.get("row_count") else None,
            column_count=int(data["column_count"])
            if data.get("column_count")
            else None,
            columns=json.loads(data["columns"]) if data.get("columns") else None,
            model_comparison=json.loads(data["model_comparison"])
            if data.get("model_comparison")
            else None,
        )

    def get_pipeline_state_dict(self) -> Dict[str, Any]:
        """Get pipeline state as a dictionary for API responses."""
        return {
            "has_data": self.has_data,
            "has_problem": self.has_problem,
            "has_features": self.has_features,
            "has_preprocessed": self.has_preprocessed,
            "has_model": self.has_model,
        }

    def update_last_accessed(self):
        """Update the last_accessed timestamp."""
        self.last_accessed = datetime.utcnow()
