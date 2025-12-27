"""
Search routes for Template Heaven API.

This module provides search functionality for templates and repositories.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request

from ...core.models import APIResponse
from ...services.template_service import template_service
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=APIResponse)
async def search_templates(
    query: str = Query(..., description="Search query"),
    stack: Optional[str] = Query(None, description="Filter by stack"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    request: Request = None
):
    """Search templates by query string."""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        templates = await template_service.search_templates(
            query=query,
            stack=stack,
            tags=tags,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            message=f"Found {len(templates)} templates for '{query}'",
            data={
                "query": query,
                "templates": [template.dict() for template in templates],
                "total_count": len(templates)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search templates"
        )


@router.get("/search/trending", response_model=APIResponse)
async def get_trending_templates(
    stack: Optional[str] = Query(None, description="Filter by stack"),
    timeframe: str = Query("week", description="Trending timeframe", regex=r"^(day|week|month)$"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100),
    request: Request = None
):
    """Get trending templates based on recent activity."""
    try:
        # For now, return templates sorted by quality score and stars
        # In a real implementation, this would analyze recent downloads, stars, etc.
        templates, total_count = await template_service.list_templates(
            stack=stack,
            limit=limit,
            sort_by="quality_score",
            sort_order="desc"
        )
        
        return APIResponse(
            success=True,
            message=f"Found {len(templates)} trending templates",
            data={
                "timeframe": timeframe,
                "templates": [template.dict() for template in templates],
                "total_count": len(templates)
            }
        )
    except Exception as e:
        logger.error(f"Error getting trending templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trending templates"
        )


@router.get("/search/suggestions", response_model=APIResponse)
async def get_search_suggestions(
    q: str = Query(..., description="Partial search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum suggestions"),
    request: Request = None
):
    """Get search suggestions based on partial query."""
    try:
        if len(q.strip()) < 2:
            return APIResponse(
                success=True,
                message="Query too short for suggestions",
                data={"suggestions": []}
            )
        
        # Get templates that match the partial query
        templates = await template_service.search_templates(
            query=q,
            limit=limit
        )
        
        # Extract suggestions from template names, technologies, and tags
        suggestions = set()
        for template in templates:
            suggestions.add(template.name)
            suggestions.update(template.technologies or [])
            suggestions.update(template.tags or [])
        
        # Convert to list and limit
        suggestions_list = list(suggestions)[:limit]
        
        return APIResponse(
            success=True,
            message=f"Found {len(suggestions_list)} suggestions",
            data={"suggestions": suggestions_list}
        )
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search suggestions"
        )
