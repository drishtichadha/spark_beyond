"""
Insights routes - Feature importance, SHAP, lift/support analysis
"""
from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request, Depends

from backend.schemas.api_models import APIResponse
from backend.services.spark_service import spark_service, get_spark_service, SparkService

router = APIRouter(prefix="/api/insights", tags=["insights"])


async def get_session_service(request: Request) -> SparkService:
    """Dependency to get session-aware SparkService."""
    session_id = getattr(request.state, "session_id", None)
    session_manager = getattr(request.app.state, "session_manager", None)

    if session_id and session_manager:
        return await get_spark_service(session_id, session_manager)

    return spark_service


@router.get("/feature-importance")
async def get_feature_importance(
    top_n: int = Query(20, ge=5, le=100),
    svc: SparkService = Depends(get_session_service)
):
    """Get feature importance from trained model"""
    try:
        importance = svc.get_feature_importance(top_n=top_n)
        return APIResponse(
            success=True,
            message=f"Top {len(importance)} features retrieved",
            data=importance
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability-impact")
async def get_probability_impact(
    top_n: int = Query(20, ge=5, le=100),
    svc: SparkService = Depends(get_session_service)
):
    """Get probability impact analysis"""
    try:
        impact = svc.get_probability_impact(top_n=top_n)
        return APIResponse(
            success=True,
            message=f"Top {len(impact)} impacts retrieved",
            data=impact
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lift-support")
async def get_lift_support_analysis(
    min_support: float = Query(0.01, ge=0.001, le=0.5),
    min_lift: float = Query(1.1, ge=1.0, le=10.0),
    max_depth: int = Query(3, ge=2, le=5),
    top_n_features: int = Query(50, ge=10, le=100),
    max_microsegments: int = Query(100, ge=10, le=500),
    svc: SparkService = Depends(get_session_service)
):
    """
    Get feature insights with lift, support, and RIG.

    This endpoint uses server-side caching - the expensive analysis is computed
    once and then filtered based on min_support and min_lift parameters.
    Subsequent calls with different thresholds are fast (filtered from cache).

    Microsegment parameters (require cache invalidation if changed):
    - max_depth: Maximum number of conditions to combine (2-5)
    - top_n_features: Number of top features to consider for combinations
    - max_microsegments: Maximum number of microsegments to return
    """
    try:
        insights = svc.get_insight_analysis(
            min_support=min_support,
            min_lift=min_lift,
            max_depth=max_depth,
            top_n_features=top_n_features,
            max_microsegments=max_microsegments
        )
        # Save insights to session cache
        await svc.save_state()
        return APIResponse(
            success=True,
            message=f"Found {len(insights['insights'])} insights (filtered from cache)",
            data=insights
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/microsegments")
async def get_microsegments_paginated(
    min_support: float = Query(0.01, ge=0.001, le=0.5),
    min_lift: float = Query(1.1, ge=1.0, le=10.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    sort_by: Literal['lift', 'support', 'rig', 'support_count'] = Query('lift'),
    sort_order: Literal['asc', 'desc'] = Query('desc'),
    svc: SparkService = Depends(get_session_service)
):
    """
    Get paginated microsegments with sorting.

    This endpoint provides lazy loading for microsegments - fetch them in chunks
    as the user scrolls. Requires that lift-support analysis has been run first.

    Args:
        min_support: Minimum support threshold for filtering (0.001-0.5)
        min_lift: Minimum lift threshold for filtering (1.0-10.0)
        page: Page number (1-indexed)
        page_size: Number of items per page (5-100)
        sort_by: Field to sort by (lift, support, rig, support_count)
        sort_order: Sort direction (asc, desc)

    Returns:
        Paginated microsegments with pagination metadata and sort info
    """
    try:
        result = svc.get_microsegments_paginated(
            min_support=min_support,
            min_lift=min_lift,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        pagination = result['pagination']
        return APIResponse(
            success=True,
            message=f"Page {pagination['page']} of {pagination['total_pages']} ({pagination['total']} total)",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shap")
async def get_shap_analysis(
    sample_size: int = Query(500, ge=100, le=2000),
    svc: SparkService = Depends(get_session_service)
):
    """Get SHAP analysis results"""
    try:
        shap_results = svc.get_shap_analysis(sample_size=sample_size)
        return APIResponse(
            success=True,
            message="SHAP analysis complete",
            data=shap_results
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
