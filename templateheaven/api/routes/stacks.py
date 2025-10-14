"""
Stack management routes for Template Heaven API.

This module provides endpoints for managing technology stacks,
configurations, and stack-specific operations.
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from fastapi.responses import JSONResponse

from ...core.models import (
    APIResponse, StackCategory, StackConfiguration, 
    PaginatedResponse
)
from ..dependencies import (
    get_settings, require_auth, require_admin, get_optional_user,
    get_request_id, get_service
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/stacks", response_model=List[StackConfiguration])
async def list_stacks(
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    List all available technology stacks.
    
    Returns a list of all configured technology stacks with their
    configurations, requirements, and metadata.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Get all stacks
        stacks = await stack_service.list_stacks()
        
        return stacks
        
    except Exception as e:
        logger.error(f"Failed to list stacks: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stacks"
        )


@router.get("/stacks/{stack_name}", response_model=StackConfiguration)
async def get_stack(
    stack_name: StackCategory = Path(..., description="Stack name"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get a specific stack configuration.
    
    Returns detailed configuration for a specific technology stack
    including requirements, quality standards, and supported technologies.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Get stack configuration
        stack = await stack_service.get_stack(stack_name)
        
        if not stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        return stack
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stack configuration"
        )


@router.get("/stacks/{stack_name}/templates", response_model=PaginatedResponse)
async def get_stack_templates(
    stack_name: StackCategory = Path(..., description="Stack name"),
    status: Optional[str] = Query(None, description="Filter by template status"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    min_stars: Optional[int] = Query(None, description="Minimum GitHub stars", ge=0),
    min_quality_score: Optional[float] = Query(None, description="Minimum quality score", ge=0.0, le=1.0),
    page: int = Query(1, description="Page number", ge=1),
    per_page: int = Query(20, description="Items per page", ge=1, le=100),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get templates for a specific stack.
    
    Returns paginated list of templates belonging to a specific
    technology stack with optional filtering.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Check if stack exists
        stack = await stack_service.get_stack(stack_name)
        if not stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        # Build filters
        filters = {"stack": stack_name}
        if status:
            filters["status"] = status
        if tags:
            filters["tags"] = tags
        if min_stars is not None:
            filters["min_stars"] = min_stars
        if min_quality_score is not None:
            filters["min_quality_score"] = min_quality_score
        
        # Get templates
        templates, total = await stack_service.get_stack_templates(
            stack_name=stack_name,
            filters=filters,
            page=page,
            per_page=per_page
        )
        
        # Calculate pagination info
        pages = (total + per_page - 1) // per_page
        has_next = page < pages
        has_prev = page > 1
        
        return PaginatedResponse(
            items=templates,
            total=total,
            page=page,
            per_page=per_page,
            pages=pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get templates for stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stack templates"
        )


@router.get("/stacks/{stack_name}/stats")
async def get_stack_stats(
    stack_name: StackCategory = Path(..., description="Stack name"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get statistics for a specific stack.
    
    Returns usage statistics, template counts, and other metrics
    for a specific technology stack.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Check if stack exists
        stack = await stack_service.get_stack(stack_name)
        if not stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        # Get stack statistics
        stats = await stack_service.get_stack_stats(stack_name)
        
        return APIResponse(
            success=True,
            message="Stack statistics retrieved",
            data=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stack statistics"
        )


@router.post("/stacks/{stack_name}/validate")
async def validate_stack_templates(
    stack_name: StackCategory = Path(..., description="Stack name"),
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Validate all templates in a stack.
    
    Runs validation checks on all templates in a specific stack
    against the stack's quality requirements.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Check if stack exists
        stack = await stack_service.get_stack(stack_name)
        if not stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        # Validate stack templates
        validation_result = await stack_service.validate_stack_templates(stack_name)
        
        return APIResponse(
            success=True,
            message="Stack validation completed",
            data=validation_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate stack templates"
        )


@router.post("/stacks", response_model=StackConfiguration, status_code=status.HTTP_201_CREATED)
async def create_stack(
    stack_data: StackConfiguration,
    current_user: Any = Depends(require_admin),
    request_id: str = Depends(get_request_id)
):
    """
    Create a new technology stack.
    
    Creates a new technology stack with custom configuration.
    Requires admin privileges.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Create stack
        stack = await stack_service.create_stack(
            stack_data=stack_data,
            created_by=current_user.id
        )
        
        logger.info(f"Stack created: {stack.name} by admin {current_user.username}")
        
        return stack
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create stack: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create stack"
        )


@router.put("/stacks/{stack_name}", response_model=StackConfiguration)
async def update_stack(
    stack_name: StackCategory = Path(..., description="Stack name"),
    stack_data: StackConfiguration = None,
    current_user: Any = Depends(require_admin),
    request_id: str = Depends(get_request_id)
):
    """
    Update a technology stack configuration.
    
    Updates the configuration for an existing technology stack.
    Requires admin privileges.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Check if stack exists
        existing_stack = await stack_service.get_stack(stack_name)
        if not existing_stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        # Update stack
        stack = await stack_service.update_stack(
            stack_name=stack_name,
            stack_data=stack_data,
            updated_by=current_user.id
        )
        
        logger.info(f"Stack updated: {stack.name} by admin {current_user.username}")
        
        return stack
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update stack"
        )


@router.delete("/stacks/{stack_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stack(
    stack_name: StackCategory = Path(..., description="Stack name"),
    current_user: Any = Depends(require_admin),
    request_id: str = Depends(get_request_id)
):
    """
    Delete a technology stack.
    
    Permanently deletes a technology stack and all its templates.
    Requires admin privileges.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Check if stack exists
        existing_stack = await stack_service.get_stack(stack_name)
        if not existing_stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        # Delete stack
        await stack_service.delete_stack(stack_name)
        
        logger.info(f"Stack deleted: {stack_name} by admin {current_user.username}")
        
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete stack"
        )


@router.get("/stacks/{stack_name}/documentation")
async def get_stack_documentation(
    stack_name: StackCategory = Path(..., description="Stack name"),
    format: str = Query("markdown", description="Documentation format", regex=r"^(markdown|html|json)$"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get stack documentation.
    
    Returns comprehensive documentation for a technology stack
    in the requested format.
    """
    try:
        # Get stack service
        stack_service = get_service("stack_service")
        
        # Check if stack exists
        stack = await stack_service.get_stack(stack_name)
        if not stack:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stack not found"
            )
        
        # Get documentation
        documentation = await stack_service.get_stack_documentation(
            stack_name=stack_name,
            format=format
        )
        
        return APIResponse(
            success=True,
            message="Stack documentation retrieved",
            data=documentation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get documentation for stack {stack_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stack documentation"
        )
