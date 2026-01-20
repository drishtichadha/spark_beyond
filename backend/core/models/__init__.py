"""
ML Models package for Spark Beyond

This package provides:
- Baseline models for comparison (Logistic Regression, Decision Tree)
- EvalML AutoML integration for automated model selection
- Model comparison framework for impact analysis
"""

from .baseline_models import BaselineModels, BaselineResult
from .evalml_runner import EvalMLRunner, AutoMLResult
from .model_comparison import ModelComparison, ComparisonResult

__all__ = [
    'BaselineModels',
    'BaselineResult',
    'EvalMLRunner',
    'AutoMLResult',
    'ModelComparison',
    'ComparisonResult'
]
