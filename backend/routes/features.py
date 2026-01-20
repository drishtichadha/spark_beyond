"""
Feature routes - Generation and preprocessing
"""
from fastapi import APIRouter, HTTPException, Request, Depends

from backend.schemas.api_models import (
    FeatureGenerationRequest, PreprocessingRequest, APIResponse, FeatureSummary
)
from backend.services.spark_service import spark_service, get_spark_service, SparkService

router = APIRouter(prefix="/api/features", tags=["features"])


async def get_session_service(request: Request) -> SparkService:
    """Dependency to get session-aware SparkService."""
    session_id = getattr(request.state, "session_id", None)
    session_manager = getattr(request.app.state, "session_manager", None)

    if session_id and session_manager:
        return await get_spark_service(session_id, session_manager)

    return spark_service


@router.post("/generate")
async def generate_features(
    request: FeatureGenerationRequest,
    svc: SparkService = Depends(get_session_service)
):
    """Generate features using AutoFeatureGenerator"""
    try:
        summary = svc.generate_features(
            include_numerical=request.include_numerical,
            include_interactions=request.include_interactions,
            include_binning=request.include_binning,
            include_datetime=request.include_datetime,
            include_string=request.include_string
        )
        # Save state after generating features
        await svc.save_state()
        return APIResponse(
            success=True,
            message=f"Generated {summary['generated_features']} new features",
            data=summary
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preprocess")
async def preprocess_features(
    request: PreprocessingRequest,
    svc: SparkService = Depends(get_session_service)
):
    """Preprocess features for model training"""
    try:
        result = svc.preprocess_features(
            imputation_strategy=request.imputation_strategy.value,
            handle_outliers=request.handle_outliers,
            outlier_strategy=request.outlier_strategy.value if request.outlier_strategy else None,
            outlier_threshold=request.outlier_threshold,
            apply_scaling=request.apply_scaling,
            scaling_strategy=request.scaling_strategy.value if request.scaling_strategy else None,
            group_rare=request.group_rare,
            rare_threshold=request.rare_threshold
        )
        # Save state after preprocessing
        await svc.save_state()
        return APIResponse(
            success=True,
            message=f"Preprocessed {result['encoded_features']} features",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_feature_summary(svc: SparkService = Depends(get_session_service)):
    """Get summary of generated features"""
    try:
        state = svc.get_state()
        if not state['has_features']:
            raise HTTPException(status_code=400, detail="Features not generated yet")

        return APIResponse(
            success=True,
            message="Feature summary retrieved",
            data={
                "has_features": state['has_features'],
                "has_preprocessed": state['has_preprocessed']
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
