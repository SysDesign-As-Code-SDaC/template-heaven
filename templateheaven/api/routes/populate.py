"""
Template population routes for Template Heaven API.

This module provides endpoints for automated template discovery,
validation, and population across technology stacks.
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse

from ...core.models import (
    APIResponse, PopulationRequest, PopulationResult, StackCategory
)
from ..dependencies import (
    get_settings, require_auth, require_admin, get_optional_user,
    get_request_id, get_service
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/populate/run", response_model=PopulationResult)
async def run_population(
    population_request: PopulationRequest,
    background_tasks: BackgroundTasks,
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Run template population process.
    
    Discovers, validates, and populates templates for specified stacks
    or all stacks. Can be run synchronously or asynchronously.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Run population
        if population_request.dry_run:
            # Dry run - just preview what would be done
            result = await population_service.preview_population(
                stack=population_request.stack,
                limit=population_request.limit,
                min_potential=population_request.min_potential
            )
        else:
            # Actual population
            result = await population_service.run_population(
                stack=population_request.stack,
                limit=population_request.limit,
                min_potential=population_request.min_potential,
                force=population_request.force,
                initiated_by=current_user.id
            )
        
        logger.info(f"Population completed by user {current_user.username}: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Population failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Population process failed"
        )


@router.post("/populate/async", status_code=status.HTTP_202_ACCEPTED)
async def run_population_async(
    population_request: PopulationRequest,
    background_tasks: BackgroundTasks,
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Run template population asynchronously.
    
    Starts the population process in the background and returns
    immediately with a job ID for tracking progress.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Start async population
        job_id = await population_service.start_async_population(
            stack=population_request.stack,
            limit=population_request.limit,
            min_potential=population_request.min_potential,
            force=population_request.force,
            initiated_by=current_user.id
        )
        
        logger.info(f"Async population started by user {current_user.username}: {job_id}")
        
        return APIResponse(
            success=True,
            message="Population process started",
            data={
                "job_id": job_id,
                "status": "started",
                "tracking_url": f"/api/v1/populate/jobs/{job_id}"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start async population: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start population process"
        )


@router.get("/populate/status")
async def get_population_status(
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get population status across all stacks.
    
    Returns the current status of template population for all
    technology stacks including counts and last update times.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Get status
        status = await population_service.get_population_status()
        
        return APIResponse(
            success=True,
            message="Population status retrieved",
            data=status
        )
        
    except Exception as e:
        logger.error(f"Failed to get population status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve population status"
        )


@router.get("/populate/status/{stack_name}")
async def get_stack_population_status(
    stack_name: StackCategory = Path(..., description="Stack name"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get population status for a specific stack.
    
    Returns detailed population status for a specific technology
    stack including template counts and quality metrics.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Get stack status
        status = await population_service.get_stack_population_status(stack_name)
        
        return APIResponse(
            success=True,
            message=f"Population status for {stack_name} retrieved",
            data=status
        )
        
    except Exception as e:
        logger.error(f"Failed to get population status for {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stack population status"
        )


@router.get("/populate/discover/{stack_name}")
async def discover_templates(
    stack_name: StackCategory = Path(..., description="Stack name"),
    limit: int = Query(10, description="Maximum candidates to discover", ge=1, le=50),
    min_potential: float = Query(0.7, description="Minimum template potential", ge=0.0, le=1.0),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Discover potential templates for a stack.
    
    Searches for potential template candidates from external sources
    (GitHub) for a specific technology stack without adding them.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Discover templates
        candidates = await population_service.discover_templates(
            stack_name=stack_name,
            limit=limit,
            min_potential=min_potential
        )
        
        return APIResponse(
            success=True,
            message=f"Template discovery completed for {stack_name}",
            data={
                "stack": stack_name,
                "candidates": candidates,
                "count": len(candidates),
                "min_potential": min_potential
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to discover templates for {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to discover templates"
        )


@router.get("/populate/jobs/{job_id}")
async def get_population_job_status(
    job_id: str = Path(..., description="Population job ID"),
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Get status of an async population job.
    
    Returns the current status and progress of an asynchronous
    population job including results and any errors.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Get job status
        job_status = await population_service.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Population job not found"
            )
        
        return APIResponse(
            success=True,
            message="Population job status retrieved",
            data=job_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status"
        )


@router.delete("/populate/jobs/{job_id}")
async def cancel_population_job(
    job_id: str = Path(..., description="Population job ID"),
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Cancel an async population job.
    
    Cancels a running asynchronous population job if it's still
    in progress. Completed jobs cannot be cancelled.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Cancel job
        result = await population_service.cancel_job(job_id, current_user.id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Population job not found or cannot be cancelled"
            )
        
        logger.info(f"Population job {job_id} cancelled by user {current_user.username}")
        
        return APIResponse(
            success=True,
            message="Population job cancelled",
            data={"job_id": job_id, "status": "cancelled"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel population job"
        )


@router.get("/populate/jobs")
async def list_population_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    stack: Optional[StackCategory] = Query(None, description="Filter by stack"),
    limit: int = Query(20, description="Maximum jobs to return", ge=1, le=100),
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    List population jobs.
    
    Returns a list of population jobs with optional filtering
    by status and stack. Only shows jobs for the current user
    unless they have admin privileges.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Build filters
        filters = {}
        if status:
            filters["status"] = status
        if stack:
            filters["stack"] = stack
        
        # Get jobs
        jobs = await population_service.list_jobs(
            filters=filters,
            limit=limit,
            user_id=current_user.id,
            is_admin=current_user.role in ["admin", "moderator"]
        )
        
        return APIResponse(
            success=True,
            message="Population jobs retrieved",
            data={
                "jobs": jobs,
                "count": len(jobs),
                "filters": filters
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list population jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve population jobs"
        )


@router.post("/populate/validate/{stack_name}")
async def validate_stack_population(
    stack_name: StackCategory = Path(..., description="Stack name"),
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Validate population for a specific stack.
    
    Runs validation checks on all templates in a stack to ensure
    they meet the stack's quality requirements and standards.
    """
    try:
        # Get population service
        population_service = get_service("population_service")
        
        # Validate stack population
        validation_result = await population_service.validate_stack_population(stack_name)
        
        return APIResponse(
            success=True,
            message=f"Population validation completed for {stack_name}",
            data=validation_result
        )
        
    except Exception as e:
        logger.error(f"Failed to validate population for {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate stack population"
        )
