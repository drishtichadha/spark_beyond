"""
Data routes - Loading, quality checks, schema validation
"""
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional
import logging

from backend.schemas.api_models import (
    LoadDataRequest, ProblemDefinition, APIResponse,
    DatasetInfo, QualityReport, SchemaInfo
)
from backend.services.spark_service import spark_service, get_spark_service, SparkService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data", tags=["data"])


async def get_session_service(request: Request) -> SparkService:
    """
    Dependency to get session-aware SparkService.

    Falls back to global singleton if session middleware is not configured.
    """
    session_id = getattr(request.state, "session_id", None)
    session_manager = getattr(request.app.state, "session_manager", None)

    if session_id and session_manager:
        return await get_spark_service(session_id, session_manager)

    # Fallback to global singleton
    return spark_service


@router.get("/state")
async def get_state(svc: SparkService = Depends(get_session_service)):
    """Get current pipeline state"""
    try:
        state = svc.get_state()
        return APIResponse(success=True, message="State retrieved", data=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_data(request: LoadDataRequest, svc: SparkService = Depends(get_session_service)):
    """Load dataset from CSV file"""
    try:
        info = svc.load_data(request.file_path)
        # Save state after loading data
        await svc.save_state()
        return APIResponse(
            success=True,
            message=f"Dataset loaded: {info['rows']} rows x {info['columns']} columns",
            data=info
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/problem")
async def set_problem(request: ProblemDefinition, svc: SparkService = Depends(get_session_service)):
    """Set problem definition and validate schema"""
    logger.info(f"set_problem called with: target={request.target}, type={request.type}, desired_result={request.desired_result}, date_column={request.date_column}")
    try:
        schema_info = svc.set_problem(
            target=request.target,
            problem_type=request.type.value,
            desired_result=request.desired_result,
            date_column=request.date_column
        )
        # Save state after setting problem
        await svc.save_state()
        return APIResponse(
            success=True,
            message="Problem defined and schema validated",
            data=schema_info
        )
    except ValueError as e:
        logger.error(f"ValueError in set_problem: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in set_problem: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema")
async def get_schema(svc: SparkService = Depends(get_session_service)):
    """Get current schema information"""
    try:
        if svc.schema_checker is None:
            raise HTTPException(status_code=400, detail="Problem not defined yet")

        schema_info = svc.schema_checker.check()
        return APIResponse(success=True, message="Schema retrieved", data=schema_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-check")
async def run_quality_check(svc: SparkService = Depends(get_session_service)):
    """Run data quality checks"""
    try:
        report = svc.run_quality_check()
        return APIResponse(
            success=True,
            message=f"Quality score: {report['quality_score']}/100",
            data=report
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/columns")
async def get_columns(svc: SparkService = Depends(get_session_service)):
    """Get column information"""
    try:
        if svc.df is None:
            raise HTTPException(status_code=400, detail="No data loaded")

        info = svc.get_dataset_info()
        return APIResponse(success=True, message="Columns retrieved", data=info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_state(svc: SparkService = Depends(get_session_service)):
    """Reset all pipeline state"""
    try:
        svc.reset()
        # Save state after reset
        await svc.save_state()
        return APIResponse(success=True, message="Pipeline state reset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profile")
async def run_data_profile(
    minimal: bool = True,
    max_rows: int = 50000,
    svc: SparkService = Depends(get_session_service)
):
    """Run comprehensive data profiling using ydata-profiling"""
    try:
        logger.info(f"Running data profile with minimal={minimal}, max_rows={max_rows}")
        report = svc.run_data_profile(minimal=minimal, max_rows=max_rows)
        # Save state after profiling
        await svc.save_state()
        logger.info(f"Profile completed successfully with {len(report.get('summary', {}))} summary keys")
        return APIResponse(
            success=True,
            message="Data profile generated",
            data=report
        )
    except ValueError as e:
        logger.error(f"ValueError in run_data_profile: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in run_data_profile: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
