"""
Model routes - Training and evaluation
"""
from fastapi import APIRouter, HTTPException, Request, Depends

from backend.schemas.api_models import (
    TrainingRequest, AutoMLRequest, APIResponse, ModelResult, TrainingMetrics
)
from backend.services.spark_service import spark_service, get_spark_service, SparkService

router = APIRouter(prefix="/api/models", tags=["models"])


async def get_session_service(request: Request) -> SparkService:
    """Dependency to get session-aware SparkService."""
    session_id = getattr(request.state, "session_id", None)
    session_manager = getattr(request.app.state, "session_manager", None)

    if session_id and session_manager:
        return await get_spark_service(session_id, session_manager)

    return spark_service


@router.post("/train")
async def train_model(
    request: TrainingRequest,
    svc: SparkService = Depends(get_session_service)
):
    """Train XGBoost model"""
    try:
        result = svc.train_model(
            train_split=request.train_split,
            max_depth=request.max_depth,
            learning_rate=request.learning_rate,
            num_rounds=request.num_rounds
        )
        # Save state after training
        await svc.save_state()
        return APIResponse(
            success=True,
            message="Model trained successfully",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(svc: SparkService = Depends(get_session_service)):
    """Get current model metrics"""
    try:
        if svc.feature_selector is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")

        state = svc.get_state()
        if not state['has_model']:
            raise HTTPException(status_code=400, detail="Model not trained yet")

        return APIResponse(
            success=True,
            message="Metrics retrieved",
            data=svc._metrics
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-baselines")
async def train_baselines(svc: SparkService = Depends(get_session_service)):
    """Train baseline models for comparison"""
    try:
        from backend.core.models.baseline_models import BaselineModels

        if svc._transformed_df is None:
            raise HTTPException(status_code=400, detail="Features not preprocessed yet")

        # Split data
        train_df = svc._transformed_df.sample(0.8, seed=42)
        test_df = svc._transformed_df.sample(0.2, seed=42)

        baselines = BaselineModels(
            problem=svc.problem,
            train_df=train_df,
            test_df=test_df,
            feature_col=svc._feature_output_col,
            label_col=svc.problem.target
        )

        results = []

        # Naive baseline
        naive_result = baselines.train_naive_baseline()
        results.append({
            "model_name": naive_result.model_name,
            "metrics": naive_result.metrics,
            "training_time": naive_result.training_time
        })

        # Decision Tree
        dt_result = baselines.train_decision_tree(max_depth=5)
        results.append({
            "model_name": dt_result.model_name,
            "metrics": dt_result.metrics,
            "training_time": dt_result.training_time
        })

        # Logistic Regression (for classification)
        if svc.problem.type == "classification":
            lr_result = baselines.train_logistic_regression()
            results.append({
                "model_name": lr_result.model_name,
                "metrics": lr_result.metrics,
                "training_time": lr_result.training_time
            })

        return APIResponse(
            success=True,
            message=f"Trained {len(results)} baseline models",
            data=results
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automl")
async def run_automl(
    request: AutoMLRequest,
    svc: SparkService = Depends(get_session_service)
):
    """Run AutoML search"""
    try:
        from backend.core.models.evalml_runner import AutoMLRunner

        if svc.df is None:
            raise HTTPException(status_code=400, detail="No data loaded")

        df_to_use = svc._df_with_features if svc._df_with_features else svc.df

        runner = AutoMLRunner(
            spark=svc.spark,
            problem=svc.problem,
            max_rows_for_pandas=50000,
            verbose=True
        )

        result = runner.run_automl(
            spark_df=df_to_use,
            timeout=request.timeout,
            cpu_limit=request.cpu_limit,
            quick_mode=request.quick_mode
        )

        return APIResponse(
            success=True,
            message=f"AutoML completed in {result.search_time:.1f}s",
            data={
                "best_score": result.best_score,
                "problem_type": result.problem_type,
                "metric": result.metric,
                "search_time": result.search_time,
                "model_summary": result.model_summary,
                "feature_importance": result.feature_importance.head(15).to_dict('records') if result.feature_importance is not None else None
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-features")
async def compare_base_vs_engineered(svc: SparkService = Depends(get_session_service)):
    """
    Compare model performance with base features vs engineered features.
    Trains models on both original and engineered feature sets.
    """
    try:
        result = svc.compare_base_vs_engineered_features()
        # Save state after comparison
        await svc.save_state()
        return APIResponse(
            success=True,
            message="Feature comparison completed",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison-summary")
async def get_comparison_summary(svc: SparkService = Depends(get_session_service)):
    """Get the current model comparison summary"""
    try:
        summary = svc.get_model_comparison_summary()
        return APIResponse(
            success=True,
            message="Comparison summary retrieved",
            data=summary
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
