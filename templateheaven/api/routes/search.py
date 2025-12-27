"""
Search routes for Template Heaven API.

This module provides search functionality for templates across
local and external sources with advanced filtering and ranking.
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse

from ...core.models import (
    APIResponse, SearchRequest, TemplateSearchResult, 
    StackCategory, PaginatedResponse
)
from ..dependencies import (
    get_settings, get_optional_user, get_request_id, get_service
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=PaginatedResponse)
async def search_templates(
    search_request: SearchRequest,
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Search templates with advanced filtering.
    
    Performs comprehensive search across local templates and external
    sources (GitHub) with relevance scoring and filtering.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Perform search
        results, total = await search_service.search_templates(
            query=search_request.query,
            stack=search_request.stack,
            tags=search_request.tags,
            min_stars=search_request.min_stars,
            min_quality_score=search_request.min_quality_score,
            include_external=search_request.include_external,
            limit=search_request.limit,
            page=search_request.page
        )
        
        # Calculate pagination info
        per_page = search_request.limit
        pages = (total + per_page - 1) // per_page
        has_next = search_request.page < pages
        has_prev = search_request.page > 1
        
        return PaginatedResponse(
            items=results,
            total=total,
            page=search_request.page,
            per_page=per_page,
            pages=pages,
            has_next=has_next,
            has_prev=has_prev
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )


@router.get("/search/quick")
async def quick_search(
    q: str = Query(..., description="Search query", min_length=1, max_length=200),
    stack: Optional[StackCategory] = Query(None, description="Filter by stack"),
    limit: int = Query(10, description="Maximum results", ge=1, le=50),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Quick search for templates.
    
    Fast search endpoint for autocomplete and quick suggestions.
    Returns top results with minimal metadata.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Perform quick search
        results = await search_service.quick_search(
            query=q,
            stack=stack,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            message="Quick search completed",
            data={
                "query": q,
                "results": results,
                "count": len(results)
            }
        )
        
    except Exception as e:
        logger.error(f"Quick search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quick search failed"
        )


@router.get("/search/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial query", min_length=1, max_length=100),
    stack: Optional[StackCategory] = Query(None, description="Filter by stack"),
    limit: int = Query(5, description="Maximum suggestions", ge=1, le=20),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get search suggestions.
    
    Returns autocomplete suggestions based on partial query input.
    Useful for search UI components.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Get suggestions
        suggestions = await search_service.get_suggestions(
            query=q,
            stack=stack,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            message="Search suggestions retrieved",
            data={
                "query": q,
                "suggestions": suggestions
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve search suggestions"
        )


@router.get("/search/trending")
async def get_trending_templates(
    stack: Optional[StackCategory] = Query(None, description="Filter by stack"),
    timeframe: str = Query("week", description="Trending timeframe", regex=r"^(day|week|month)$"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get trending templates.
    
    Returns templates that are currently trending based on
    GitHub activity, downloads, and other metrics.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Get trending templates
        trending = await search_service.get_trending_templates(
            stack=stack,
            timeframe=timeframe,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            message="Trending templates retrieved",
            data={
                "timeframe": timeframe,
                "stack": stack,
                "templates": trending,
                "count": len(trending)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get trending templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trending templates"
        )


@router.get("/search/popular")
async def get_popular_templates(
    stack: Optional[StackCategory] = Query(None, description="Filter by stack"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get popular templates.
    
    Returns the most popular templates based on stars, downloads,
    and usage metrics.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Get popular templates
        popular = await search_service.get_popular_templates(
            stack=stack,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            message="Popular templates retrieved",
            data={
                "stack": stack,
                "templates": popular,
                "count": len(popular)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get popular templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve popular templates"
        )


@router.get("/search/recent")
async def get_recent_templates(
    stack: Optional[StackCategory] = Query(None, description="Filter by stack"),
    limit: int = Query(20, description="Maximum results", ge=1, le=100),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get recently added templates.
    
    Returns templates that were recently added to the system,
    ordered by creation date.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Get recent templates
        recent = await search_service.get_recent_templates(
            stack=stack,
            limit=limit
        )
        
        return APIResponse(
            success=True,
            message="Recent templates retrieved",
            data={
                "stack": stack,
                "templates": recent,
                "count": len(recent)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get recent templates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent templates"
        )


@router.get("/search/stats")
async def get_search_stats(
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Get search statistics.
    
    Returns search usage statistics, popular queries, and
    performance metrics.
    """
    try:
        # Get search service
        search_service = get_service("search_service")
        
        # Get search statistics
        stats = await search_service.get_search_stats()
        
        return APIResponse(
            success=True,
            message="Search statistics retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get search stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve search statistics"
        )
