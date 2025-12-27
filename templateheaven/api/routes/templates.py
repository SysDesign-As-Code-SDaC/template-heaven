"""
Template management routes for Template Heaven API.

This module provides REST endpoints for template CRUD operations,
search, and management functionality.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from fastapi.responses import JSONResponse

from ...core.models import (
    Template, TemplateStatus, StackCategory, APIResponse, 
    PaginatedResponse, TemplateCreateRequest, TemplateUpdateRequest
)
from ..dependencies import (
    get_settings, require_auth, require_admin, get_optional_user,
    get_request_id, get_service
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/templates", response_model=PaginatedResponse)
async def list_templates(
    stack: Optional[StackCategory] = Query(None, description="Filter by technology stack"),
    status: Optional[TemplateStatus] = Query(None, description="Filter by template status"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    min_stars: Optional[int] = Query(None, description="Minimum GitHub stars", ge=0),
    min_quality_score: Optional[float] = Query(None, description="Minimum quality score", ge=0.0, le=1.0),
    page: int = Query(1, description="Page number", ge=1),
    per_page: int = Query(20, description="Items per page", ge=1, le=100),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    List templates with filtering and pagination.
    
    Returns a paginated list of templates with optional filtering by
    stack, status, tags, and quality criteria.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Build filters
        filters = {}
        if stack:
            filters["stack"] = stack
        if status:
            filters["status"] = status
        if tags:
            filters["tags"] = tags
        if min_stars is not None:
            filters["min_stars"] = min_stars
        if min_quality_score is not None:
            filters["min_quality_score"] = min_quality_score
        
        # Get templates
        templates, total = await template_service.list_templates(
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
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve templates"
        )


@router.get("/templates/{template_id}", response_model=Template)
async def get_template(
    template_id: str = Path(..., description="Template ID"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get a specific template by ID.
    
    Returns detailed information about a single template including
    metadata, dependencies, and quality metrics.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Get template
        template = await template_service.get_template(template_id)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve template"
        )


@router.post("/templates", response_model=Template, status_code=status.HTTP_201_CREATED)
async def create_template(
    template_data: TemplateCreateRequest,
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Create a new template.
    
    Creates a new template with the provided metadata. Requires
    authentication and appropriate permissions.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Create template
        template = await template_service.create_template(
            template_data=template_data,
            created_by=current_user.id
        )
        
        logger.info(f"Template created: {template.name} by user {current_user.username}")
        
        return template
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to create template: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create template"
        )


@router.put("/templates/{template_id}", response_model=Template)
async def update_template(
    template_id: str = Path(..., description="Template ID"),
    template_data: TemplateUpdateRequest = None,
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Update an existing template.
    
    Updates template metadata. Requires authentication and ownership
    or admin privileges.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Check if template exists
        existing_template = await template_service.get_template(template_id)
        if not existing_template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Check permissions
        if (existing_template.author != current_user.username and 
            current_user.role not in ["admin", "moderator"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to update this template"
            )
        
        # Update template
        template = await template_service.update_template(
            template_id=template_id,
            template_data=template_data,
            updated_by=current_user.id
        )
        
        logger.info(f"Template updated: {template.name} by user {current_user.username}")
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template {template_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update template"
        )


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: str = Path(..., description="Template ID"),
    current_user: Any = Depends(require_admin),
    request_id: str = Depends(get_request_id)
):
    """
    Delete a template.
    
    Permanently deletes a template. Requires admin privileges.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Check if template exists
        existing_template = await template_service.get_template(template_id)
        if not existing_template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Delete template
        await template_service.delete_template(template_id)
        
        logger.info(f"Template deleted: {template_id} by admin {current_user.username}")
        
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {template_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete template"
        )


@router.get("/templates/{template_id}/download")
async def download_template(
    template_id: str = Path(..., description="Template ID"),
    format: str = Query("zip", description="Download format", regex=r"^(zip|tar\.gz)$"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Download a template as an archive.
    
    Downloads the template files as a ZIP or TAR.GZ archive.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Check if template exists
        template = await template_service.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Generate download URL or stream
        download_info = await template_service.get_download_info(
            template_id=template_id,
            format=format
        )
        
        return download_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download template {template_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to prepare template download"
        )


@router.post("/templates/{template_id}/validate")
async def validate_template(
    template_id: str = Path(..., description="Template ID"),
    current_user: Any = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Validate a template.
    
    Runs validation checks on a template including quality assessment,
    dependency verification, and best practices compliance.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Check if template exists
        template = await template_service.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Run validation
        validation_result = await template_service.validate_template(template_id)
        
        return APIResponse(
            success=True,
            message="Template validation completed",
            data=validation_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate template {template_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate template"
        )


@router.get("/templates/{template_id}/stats")
async def get_template_stats(
    template_id: str = Path(..., description="Template ID"),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get template statistics.
    
    Returns usage statistics, download counts, and other metrics
    for a specific template.
    """
    try:
        # Get template service
        template_service = get_service("template_service")
        
        # Check if template exists
        template = await template_service.get_template(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        # Get statistics
        stats = await template_service.get_template_stats(template_id)
        
        return APIResponse(
            success=True,
            message="Template statistics retrieved",
            data=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template stats {template_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve template statistics"
        )
