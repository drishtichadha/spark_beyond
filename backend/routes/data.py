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
from backend.core.utils.path_validator import validate_file_path, PathValidationError
from backend.config import get_resource_limits, DEFAULT_LIMITS

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


@router.get("/limits")
async def get_limits():
    """
    Get resource limits for the current tier.

    Returns limits for:
    - File size
    - Row count
    - Feature generation
    - Training time
    - Data profiling
    """
    try:
        limits = DEFAULT_LIMITS
        return APIResponse(
            success=True,
            message="Resource limits retrieved",
            data=limits.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_data(request: LoadDataRequest, svc: SparkService = Depends(get_session_service)):
    """
    Load dataset from CSV file.

    Security:
    - File path is validated to prevent path traversal attacks.
    - File size is checked before loading to prevent resource exhaustion.
    - Row count is limited to prevent memory issues.

    Limits are enforced at multiple layers:
    1. File size check before reading
    2. Row count limit during loading
    3. Resource limits based on tier (production/development/enterprise)
    """
    try:
        import os

        # Get resource limits
        limits = DEFAULT_LIMITS

        # Validate file path to prevent path traversal attacks
        validated_path = validate_file_path(request.file_path)
        logger.info(f"Loading data from validated path: {validated_path}")

        # Check if file exists
        if not os.path.exists(validated_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

        # Check file size before loading
        file_size_bytes = os.path.getsize(validated_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Enforce max file size (use request value if provided, otherwise use tier limit)
        max_size_mb = min(
            request.max_file_size_mb or limits.max_file_size_mb,
            limits.max_file_size_mb  # Tier limit is the hard cap
        )

        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.1f}MB exceeds maximum of {max_size_mb}MB. "
                       f"Please use a smaller dataset or contact support for higher limits."
            )

        logger.info(f"File size: {file_size_mb:.1f}MB (limit: {max_size_mb}MB)")

        # Enforce max rows (use request value if provided, otherwise use tier limit)
        max_rows = min(
            request.max_rows or limits.default_max_rows,
            limits.absolute_max_rows  # Absolute hard limit
        )

        # Load data using validated path with row limit
        info = svc.load_data(str(validated_path), max_rows=max_rows)

        # Check if data was truncated
        if info['rows'] >= max_rows:
            logger.warning(f"Dataset truncated to {max_rows} rows")
            return APIResponse(
                success=True,
                message=f"Dataset loaded (truncated): {info['rows']} rows x {info['columns']} columns. "
                        f"Original file may contain more rows.",
                data={**info, "truncated": True, "max_rows": max_rows}
            )

        # Save state after loading data
        await svc.save_state()
        return APIResponse(
            success=True,
            message=f"Dataset loaded: {info['rows']} rows x {info['columns']} columns",
            data={**info, "truncated": False}
        )
    except PathValidationError as e:
        logger.warning(f"Path validation failed: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.warning(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
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
    """
    Run comprehensive data profiling using ydata-profiling.

    Args:
        minimal: Use minimal profiling for faster results
        max_rows: Maximum rows to sample (enforced: 1-100,000)
    """
    try:
        # Enforce max_rows limit for profiling
        max_rows = min(max(max_rows, 1), 100_000)  # Clamp between 1 and 100K

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
