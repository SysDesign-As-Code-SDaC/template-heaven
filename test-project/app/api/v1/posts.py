"""
Posts API endpoints.

This module provides post management endpoints including
CRUD operations for posts and content management.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from pydantic import BaseModel, Field
from datetime import datetime

from app.core.database import get_db
from app.core.models import User, Post
from app.core.exceptions import ValidationError, NotFoundError, AuthorizationError
from app.api.dependencies import (
    get_current_user, get_optional_current_user, get_pagination_params
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Pydantic models
class PostResponse(BaseModel):
    """Post response model."""
    id: int
    title: str
    content: str
    summary: Optional[str] = None
    slug: Optional[str] = None
    tags: Optional[str] = None
    is_published: bool
    is_featured: bool
    view_count: int
    author_id: int
    author_name: str
    created_at: str
    updated_at: str
    published_at: Optional[str] = None
    
    class Config:
        from_attributes = True


class PostCreate(BaseModel):
    """Post creation model."""
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=10)
    summary: Optional[str] = Field(None, max_length=500)
    tags: Optional[str] = Field(None, max_length=500)
    is_published: bool = False
    is_featured: bool = False


class PostUpdate(BaseModel):
    """Post update model."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = Field(None, min_length=10)
    summary: Optional[str] = Field(None, max_length=500)
    tags: Optional[str] = Field(None, max_length=500)
    is_published: Optional[bool] = None
    is_featured: Optional[bool] = None


class PostListResponse(BaseModel):
    """Post list response model."""
    posts: List[PostResponse]
    total: int
    page: int
    size: int
    pages: int


# Post management endpoints
@router.get("/", response_model=PostListResponse, summary="List posts")
async def list_posts(
    pagination: Dict[str, int] = Depends(get_pagination_params),
    search: Optional[str] = Query(None, description="Search in title and content"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    author_id: Optional[int] = Query(None, description="Filter by author ID"),
    is_published: Optional[bool] = Query(True, description="Filter by published status"),
    is_featured: Optional[bool] = Query(None, description="Filter by featured status"),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db)
) -> PostListResponse:
    """
    List posts with pagination and filtering.
    
    Args:
        pagination: Pagination parameters
        search: Search query
        tags: Filter by tags
        author_id: Filter by author ID
        is_published: Filter by published status
        is_featured: Filter by featured status
        current_user: Current user (if authenticated)
        db: Database session
        
    Returns:
        Paginated list of posts
    """
    # Build query with joins
    query = select(Post, User.full_name.label('author_name')).join(User, Post.author_id == User.id)
    
    # Apply filters
    if search:
        query = query.where(
            or_(
                Post.title.ilike(f"%{search}%"),
                Post.content.ilike(f"%{search}%"),
                Post.summary.ilike(f"%{search}%")
            )
        )
    
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            query = query.where(Post.tags.ilike(f"%{tag}%"))
    
    if author_id:
        query = query.where(Post.author_id == author_id)
    
    if is_published is not None:
        query = query.where(Post.is_published == is_published)
    
    if is_featured is not None:
        query = query.where(Post.is_featured == is_featured)
    
    # If user is not authenticated or not the author, only show published posts
    if not current_user:
        query = query.where(Post.is_published == True)
    elif not current_user.is_superuser:
        query = query.where(
            or_(
                Post.is_published == True,
                Post.author_id == current_user.id
            )
        )
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination and ordering
    query = query.offset(pagination["offset"]).limit(pagination["size"])
    query = query.order_by(Post.created_at.desc())
    
    # Execute query
    result = await db.execute(query)
    rows = result.all()
    
    # Convert to response format
    posts = []
    for row in rows:
        post_data = row.Post.__dict__.copy()
        post_data['author_name'] = row.author_name
        posts.append(PostResponse(**post_data))
    
    # Calculate pages
    pages = (total + pagination["size"] - 1) // pagination["size"]
    
    return PostListResponse(
        posts=posts,
        total=total,
        page=pagination["page"],
        size=pagination["size"],
        pages=pages
    )


@router.get("/{post_id}", response_model=PostResponse, summary="Get post by ID")
async def get_post(
    post_id: int,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db)
) -> PostResponse:
    """
    Get post by ID.
    
    Args:
        post_id: Post ID
        current_user: Current user (if authenticated)
        db: Database session
        
    Returns:
        Post information
        
    Raises:
        NotFoundError: If post is not found
        AuthorizationError: If user doesn't have access
    """
    # Build query with join
    query = select(Post, User.full_name.label('author_name')).join(
        User, Post.author_id == User.id
    ).where(Post.id == post_id)
    
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        raise NotFoundError("Post not found")
    
    post = row.Post
    author_name = row.author_name
    
    # Check access permissions
    if not post.is_published:
        if not current_user:
            raise AuthorizationError("Authentication required to view unpublished posts")
        if not current_user.is_superuser and post.author_id != current_user.id:
            raise AuthorizationError("Access denied to unpublished post")
    
    # Increment view count
    await db.execute(
        update(Post)
        .where(Post.id == post_id)
        .values(view_count=Post.view_count + 1)
    )
    await db.commit()
    
    # Convert to response format
    post_data = post.__dict__.copy()
    post_data['author_name'] = author_name
    
    return PostResponse(**post_data)


@router.post("/", response_model=PostResponse, summary="Create new post")
async def create_post(
    post_data: PostCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PostResponse:
    """
    Create a new post.
    
    Args:
        post_data: Post creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created post information
    """
    # Generate slug from title
    slug = post_data.title.lower().replace(" ", "-").replace("_", "-")
    # Remove special characters except hyphens
    import re
    slug = re.sub(r'[^a-z0-9\-]', '', slug)
    # Remove multiple consecutive hyphens
    slug = re.sub(r'-+', '-', slug).strip('-')
    
    # Ensure slug is unique
    counter = 1
    original_slug = slug
    while True:
        result = await db.execute(
            select(Post).where(Post.slug == slug)
        )
        existing_post = result.scalar_one_or_none()
        
        if not existing_post:
            break
        
        slug = f"{original_slug}-{counter}"
        counter += 1
    
    # Create new post
    new_post = Post(
        title=post_data.title,
        content=post_data.content,
        summary=post_data.summary,
        tags=post_data.tags,
        slug=slug,
        is_published=post_data.is_published,
        is_featured=post_data.is_featured,
        author_id=current_user.id,
        published_at=datetime.utcnow() if post_data.is_published else None
    )
    
    db.add(new_post)
    await db.commit()
    await db.refresh(new_post)
    
    logger.info(f"User {current_user.email} created post: {new_post.title}")
    
    # Get author name for response
    result = await db.execute(
        select(User.full_name).where(User.id == current_user.id)
    )
    author_name = result.scalar()
    
    # Convert to response format
    post_data_dict = new_post.__dict__.copy()
    post_data_dict['author_name'] = author_name
    
    return PostResponse(**post_data_dict)


@router.put("/{post_id}", response_model=PostResponse, summary="Update post")
async def update_post(
    post_id: int,
    post_data: PostUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> PostResponse:
    """
    Update post.
    
    Args:
        post_id: Post ID
        post_data: Post update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated post information
        
    Raises:
        NotFoundError: If post is not found
        AuthorizationError: If user doesn't have permission
    """
    # Get post
    result = await db.execute(
        select(Post).where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    
    if not post:
        raise NotFoundError("Post not found")
    
    # Check permissions
    if not current_user.is_superuser and post.author_id != current_user.id:
        raise AuthorizationError("You can only edit your own posts")
    
    # Update post
    update_data = post_data.dict(exclude_unset=True)
    
    # Handle publication status change
    if 'is_published' in update_data:
        if update_data['is_published'] and not post.is_published:
            update_data['published_at'] = datetime.utcnow()
        elif not update_data['is_published'] and post.is_published:
            update_data['published_at'] = None
    
    # Update slug if title changed
    if 'title' in update_data:
        slug = update_data['title'].lower().replace(" ", "-").replace("_", "-")
        import re
        slug = re.sub(r'[^a-z0-9\-]', '', slug)
        slug = re.sub(r'-+', '-', slug).strip('-')
        
        # Ensure slug is unique (excluding current post)
        counter = 1
        original_slug = slug
        while True:
            result = await db.execute(
                select(Post).where(and_(Post.slug == slug, Post.id != post_id))
            )
            existing_post = result.scalar_one_or_none()
            
            if not existing_post:
                break
            
            slug = f"{original_slug}-{counter}"
            counter += 1
        
        update_data['slug'] = slug
    
    if update_data:
        await db.execute(
            update(Post)
            .where(Post.id == post_id)
            .values(**update_data)
        )
        await db.commit()
        await db.refresh(post)
    
    logger.info(f"User {current_user.email} updated post: {post.title}")
    
    # Get author name for response
    result = await db.execute(
        select(User.full_name).where(User.id == post.author_id)
    )
    author_name = result.scalar()
    
    # Convert to response format
    post_data_dict = post.__dict__.copy()
    post_data_dict['author_name'] = author_name
    
    return PostResponse(**post_data_dict)


@router.delete("/{post_id}", summary="Delete post")
async def delete_post(
    post_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete post.
    
    Args:
        post_id: Post ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        NotFoundError: If post is not found
        AuthorizationError: If user doesn't have permission
    """
    # Get post
    result = await db.execute(
        select(Post).where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    
    if not post:
        raise NotFoundError("Post not found")
    
    # Check permissions
    if not current_user.is_superuser and post.author_id != current_user.id:
        raise AuthorizationError("You can only delete your own posts")
    
    # Delete post
    await db.execute(
        delete(Post).where(Post.id == post_id)
    )
    await db.commit()
    
    logger.info(f"User {current_user.email} deleted post: {post.title}")
    
    return {"message": "Post deleted successfully"}


@router.post("/{post_id}/publish", summary="Publish post")
async def publish_post(
    post_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Publish a post.
    
    Args:
        post_id: Post ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        NotFoundError: If post is not found
        AuthorizationError: If user doesn't have permission
    """
    # Get post
    result = await db.execute(
        select(Post).where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    
    if not post:
        raise NotFoundError("Post not found")
    
    # Check permissions
    if not current_user.is_superuser and post.author_id != current_user.id:
        raise AuthorizationError("You can only publish your own posts")
    
    # Publish post
    await db.execute(
        update(Post)
        .where(Post.id == post_id)
        .values(
            is_published=True,
            published_at=datetime.utcnow()
        )
    )
    await db.commit()
    
    logger.info(f"User {current_user.email} published post: {post.title}")
    
    return {"message": "Post published successfully"}


@router.post("/{post_id}/unpublish", summary="Unpublish post")
async def unpublish_post(
    post_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Unpublish a post.
    
    Args:
        post_id: Post ID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        NotFoundError: If post is not found
        AuthorizationError: If user doesn't have permission
    """
    # Get post
    result = await db.execute(
        select(Post).where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    
    if not post:
        raise NotFoundError("Post not found")
    
    # Check permissions
    if not current_user.is_superuser and post.author_id != current_user.id:
        raise AuthorizationError("You can only unpublish your own posts")
    
    # Unpublish post
    await db.execute(
        update(Post)
        .where(Post.id == post_id)
        .values(
            is_published=False,
            published_at=None
        )
    )
    await db.commit()
    
    logger.info(f"User {current_user.email} unpublished post: {post.title}")
    
    return {"message": "Post unpublished successfully"}


@router.get("/slug/{slug}", response_model=PostResponse, summary="Get post by slug")
async def get_post_by_slug(
    slug: str,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db)
) -> PostResponse:
    """
    Get post by slug.
    
    Args:
        slug: Post slug
        current_user: Current user (if authenticated)
        db: Database session
        
    Returns:
        Post information
        
    Raises:
        NotFoundError: If post is not found
        AuthorizationError: If user doesn't have access
    """
    # Build query with join
    query = select(Post, User.full_name.label('author_name')).join(
        User, Post.author_id == User.id
    ).where(Post.slug == slug)
    
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        raise NotFoundError("Post not found")
    
    post = row.Post
    author_name = row.author_name
    
    # Check access permissions
    if not post.is_published:
        if not current_user:
            raise AuthorizationError("Authentication required to view unpublished posts")
        if not current_user.is_superuser and post.author_id != current_user.id:
            raise AuthorizationError("Access denied to unpublished post")
    
    # Increment view count
    await db.execute(
        update(Post)
        .where(Post.slug == slug)
        .values(view_count=Post.view_count + 1)
    )
    await db.commit()
    
    # Convert to response format
    post_data = post.__dict__.copy()
    post_data['author_name'] = author_name
    
    return PostResponse(**post_data)
