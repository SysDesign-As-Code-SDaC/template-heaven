"""
Template management routes for Template Heaven API.

This module provides REST endpoints for template CRUD operations,
search, and management functionality.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Request
from fastapi.responses import JSONResponse, Response

from ...core.models import (
    Template, StackCategory, APIResponse
)
from ...services.template_service import template_service
from ...services.auth_service import auth_service
from ..dependencies import get_settings
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/templates", response_model=APIResponse)
async def list_templates(
    stack: Optional[str] = Query(None, description="Filter by stack"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    sort_by: str = Query("quality_score", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    request: Request = None
):
    """List templates with filtering and pagination."""
    try:
        templates, total_count = await template_service.list_templates(
            stack=stack,
            tags=tags,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return APIResponse(
            success=True,
            message=f"Found {len(templates)} templates",
            data={
                "templates": [template.dict() for template in templates],
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            }
        )
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list templates"
        )


@router.get("/templates/{template_id}", response_model=APIResponse)
async def get_template(
    template_id: str = Path(..., description="Template ID or name"),
    stack: Optional[str] = Query(None, description="Stack name (if template_id is name)"),
    request: Request = None
):
    """Get a specific template by ID or name."""
    try:
        template = await template_service.get_template(template_id, stack)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        return APIResponse(
            success=True,
            message="Template retrieved successfully",
            data={"template": template.dict()}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get template"
        )


@router.post("/templates", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    template_data: Dict[str, Any],
    request: Request = None
):
    """Create a new template."""
    try:
        # Validate required fields
        if not template_data.get("name"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Template name is required"
            )
        
        template = await template_service.create_template(template_data)
        
        return APIResponse(
            success=True,
            message="Template created successfully",
            data={"template": template.dict()}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create template"
        )


@router.put("/templates/{template_id}", response_model=APIResponse)
async def update_template(
    template_id: str = Path(..., description="Template ID"),
    update_data: Dict[str, Any] = None,
    request: Request = None
):
    """Update an existing template."""
    try:
        template = await template_service.update_template(template_id, update_data)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        return APIResponse(
            success=True,
            message="Template updated successfully",
            data={"template": template.dict()}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update template"
        )


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: str = Path(..., description="Template ID"),
    request: Request = None
):
    """Delete a template."""
    try:
        success = await template_service.delete_template(template_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete template"
        )


@router.get("/templates/{template_id}/download")
async def download_template(
    template_id: str = Path(..., description="Template ID"),
    format: str = Query("zip", description="Download format", regex=r"^(zip|tar\.gz)$"),
    request: Request = None
):
    """Download a template as an archive."""
    try:
        # Get client info
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # TODO: Get user_id from authentication when implemented
        user_id = None
        
        archive_bytes = await template_service.download_template(
            template_id=template_id,
            format=format,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not archive_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found or no files available"
            )
        
        # Return file
        filename = f"template_{template_id}.{format}"
        return Response(
            content=archive_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download template"
        )


@router.post("/templates/sync-github", response_model=APIResponse)
async def sync_from_github(
    github_url: str,
    stack: str = Query("other", description="Target stack"),
    request: Request = None
):
    """Sync a template from GitHub repository."""
    try:
        template = await template_service.sync_from_github(github_url, stack)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to sync from GitHub. Check URL and repository access."
            )
        
        return APIResponse(
            success=True,
            message="Template synced from GitHub successfully",
            data={"template": template.dict()}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing from GitHub: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync from GitHub"
        )
