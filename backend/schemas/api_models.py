"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ============ Enums ============

class ProblemType(str, Enum):
    classification = "classification"
    regression = "regression"


class ImputationStrategy(str, Enum):
    mean = "mean"
    median = "median"
    mode = "mode"
    drop = "drop"


class OutlierStrategy(str, Enum):
    iqr_cap = "iqr_cap"
    zscore_cap = "zscore_cap"
    iqr_remove = "iqr_remove"
    zscore_remove = "zscore_remove"


class ScalingStrategy(str, Enum):
    standard = "standard"
    minmax = "minmax"
    robust = "robust"
    none = "none"


# ============ Request Models ============

class LoadDataRequest(BaseModel):
    file_path: str = Field(..., description="Path to the CSV file")
    max_rows: Optional[int] = Field(
        1_000_000,
        description="Maximum number of rows to load (default: 1 million)",
        ge=1,
        le=10_000_000
    )
    max_file_size_mb: Optional[int] = Field(
        500,
        description="Maximum file size in MB (default: 500MB)",
        ge=1,
        le=2000
    )


class ProblemDefinition(BaseModel):
    target: str = Field(..., description="Target column name")
    type: ProblemType = Field(..., description="Problem type")
    desired_result: Optional[str] = Field(None, description="Desired result for classification")
    date_column: Optional[str] = Field(None, description="Date time column of the dataset")


class FeatureGenerationRequest(BaseModel):
    include_numerical: bool = True
    include_interactions: bool = True
    include_binning: bool = True
    include_datetime: bool = True
    include_string: bool = False


class PreprocessingRequest(BaseModel):
    imputation_strategy: ImputationStrategy = ImputationStrategy.median
    handle_outliers: bool = False
    outlier_strategy: Optional[OutlierStrategy] = None
    outlier_threshold: float = 1.5
    apply_scaling: bool = False
    scaling_strategy: Optional[ScalingStrategy] = None
    group_rare: bool = False
    rare_threshold: float = 0.01


class TrainingRequest(BaseModel):
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    max_depth: int = Field(4, ge=2, le=10)
    learning_rate: float = Field(0.1, ge=0.01, le=0.5)
    num_rounds: int = Field(100, ge=10, le=1000)


class AutoMLRequest(BaseModel):
    timeout: int = Field(120, ge=60, le=600)
    cpu_limit: int = Field(4, ge=1, le=8)
    quick_mode: bool = True


# ============ Response Models ============

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]


class QualityIssue(BaseModel):
    column: str
    issue: str
    severity: str


class QualityRecommendation(BaseModel):
    column: str
    action: str
    priority: str


class QualityReport(BaseModel):
    quality_score: float
    row_count: int
    column_count: int
    duplicate_count: int
    issues: List[QualityIssue]
    recommendations: List[QualityRecommendation]


class SchemaColumn(BaseModel):
    name: str
    dtype: str
    null_count: int
    distinct_count: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


class SchemaInfo(BaseModel):
    categorical: List[SchemaColumn]
    numerical: List[SchemaColumn]
    boolean: List[SchemaColumn]
    datetime: List[SchemaColumn]


class FeatureSummary(BaseModel):
    original_features: int
    total_features: int
    generated_features: int
    feature_categories: Dict[str, int]
    sample_features: Dict[str, List[str]]


class TrainingMetrics(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    r2: Optional[float] = None


class ModelResult(BaseModel):
    model_name: str
    metrics: TrainingMetrics
    training_time: float


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ProbabilityImpact(BaseModel):
    feature: str
    threshold: Optional[float]
    prob_impact: float
    left_prob: float
    right_prob: float


class InsightItem(BaseModel):
    condition: str
    lift: float
    support: float
    support_count: int
    rig: float
    class_rate: float


class Microsegment(BaseModel):
    name: str
    conditions: List[str]
    lift: float
    support: float
    support_count: int
    rig: float


class InsightAnalysisResult(BaseModel):
    target_class: str
    baseline_rate: float
    total_count: int
    insights: List[InsightItem]
    microsegments: List[Microsegment]


class SHAPResult(BaseModel):
    feature_importance: List[Dict[str, Any]]
    plot_path: Optional[str] = None


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class SessionState(BaseModel):
    has_data: bool = False
    has_problem: bool = False
    has_features: bool = False
    has_preprocessed: bool = False
    has_model: bool = False
    dataset_info: Optional[DatasetInfo] = None
    problem: Optional[ProblemDefinition] = None
