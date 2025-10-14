"""
Template service for managing templates.

This module provides business logic for template discovery, management,
validation, and operations.
"""

import asyncio
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse

import aiohttp
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database.models import Template as TemplateModel, Stack, TemplateDownload
from ..database.connection import get_db_session
from ..core.models import Template, StackCategory, APIResponse
from ..utils.logger import get_logger
from ..api.dependencies import get_settings

logger = get_logger(__name__)


class TemplateService:
    """Service for template management operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.templates_dir = Path(self.settings.templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
    
    async def list_templates(
        self,
        stack: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "quality_score",
        sort_order: str = "desc"
    ) -> Tuple[List[Template], int]:
        """
        List templates with filtering and pagination.
        
        Args:
            stack: Filter by stack name
            tags: Filter by tags
            limit: Maximum number of results
            offset: Number of results to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            Tuple of (templates, total_count)
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            # Build query
            query = select(TemplateModel).options(
                selectinload(TemplateModel.stack)
            ).where(TemplateModel.is_active == True)
            
            # Apply filters
            if stack:
                query = query.join(Stack).where(Stack.name == stack)
            
            if tags:
                # PostgreSQL JSON contains operator
                for tag in tags:
                    query = query.where(TemplateModel.tags.contains([tag]))
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(count_query)
            total_count = total_result.scalar()
            
            # Apply sorting
            if hasattr(TemplateModel, sort_by):
                sort_column = getattr(TemplateModel, sort_by)
                if sort_order.lower() == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))
            else:
                query = query.order_by(desc(TemplateModel.quality_score))
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Execute query
            result = await session.execute(query)
            template_models = result.scalars().all()
            
            # Convert to Pydantic models
            templates = []
            for template_model in template_models:
                template = self._model_to_pydantic(template_model)
                templates.append(template)
            
            return templates, total_count
    
    async def get_template(
        self, 
        template_id: str, 
        stack: Optional[str] = None
    ) -> Optional[Template]:
        """
        Get a specific template by ID or name.
        
        Args:
            template_id: Template ID or name
            stack: Stack name (if template_id is name)
            
        Returns:
            Template if found, None otherwise
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(TemplateModel).options(
                selectinload(TemplateModel.stack)
            ).where(TemplateModel.is_active == True)
            
            # Try to parse as UUID first
            try:
                import uuid
                template_uuid = uuid.UUID(template_id)
                query = query.where(TemplateModel.id == template_uuid)
            except ValueError:
                # Not a UUID, treat as name
                query = query.where(TemplateModel.name == template_id)
                if stack:
                    query = query.join(Stack).where(Stack.name == stack)
            
            result = await session.execute(query)
            template_model = result.scalar_one_or_none()
            
            if template_model:
                return self._model_to_pydantic(template_model)
            return None
    
    async def create_template(
        self, 
        template_data: Dict[str, Any], 
        user_id: Optional[str] = None
    ) -> Template:
        """
        Create a new template.
        
        Args:
            template_data: Template data
            user_id: ID of user creating the template
            
        Returns:
            Created template
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            # Get or create stack
            stack_name = template_data.get("stack", "other")
            stack_query = select(Stack).where(Stack.name == stack_name)
            stack_result = await session.execute(stack_query)
            stack = stack_result.scalar_one_or_none()
            
            if not stack:
                stack = Stack(
                    name=stack_name,
                    display_name=stack_name.title(),
                    description=f"Templates for {stack_name}",
                    technologies=template_data.get("technologies", [])
                )
                session.add(stack)
                await session.flush()
            
            # Create template model
            template_model = TemplateModel(
                name=template_data["name"],
                stack_id=stack.id,
                description=template_data.get("description"),
                path=template_data.get("path"),
                upstream_url=template_data.get("upstream_url"),
                version=template_data.get("version", "0.1.0"),
                author=template_data.get("author"),
                license=template_data.get("license"),
                tags=template_data.get("tags", []),
                technologies=template_data.get("technologies", []),
                features=template_data.get("features", []),
                dependencies=template_data.get("dependencies", {}),
                min_python_version=template_data.get("min_python_version"),
                min_node_version=template_data.get("min_node_version"),
                stars=template_data.get("stars", 0),
                forks=template_data.get("forks", 0),
                growth_rate=template_data.get("growth_rate", 0.0),
                quality_score=template_data.get("quality_score", 0.0),
                is_verified=template_data.get("is_verified", False)
            )
            
            session.add(template_model)
            await session.flush()
            
            # Load the template with stack relationship
            await session.refresh(template_model, ["stack"])
            
            return self._model_to_pydantic(template_model)
    
    async def update_template(
        self, 
        template_id: str, 
        update_data: Dict[str, Any]
    ) -> Optional[Template]:
        """
        Update an existing template.
        
        Args:
            template_id: Template ID
            update_data: Data to update
            
        Returns:
            Updated template if found, None otherwise
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            # Get template
            query = select(TemplateModel).options(
                selectinload(TemplateModel.stack)
            ).where(TemplateModel.id == template_id)
            
            result = await session.execute(query)
            template_model = result.scalar_one_or_none()
            
            if not template_model:
                return None
            
            # Update fields
            for field, value in update_data.items():
                if hasattr(template_model, field) and value is not None:
                    setattr(template_model, field, value)
            
            template_model.updated_at = datetime.utcnow()
            
            await session.flush()
            await session.refresh(template_model, ["stack"])
            
            return self._model_to_pydantic(template_model)
    
    async def delete_template(self, template_id: str) -> bool:
        """
        Delete a template (soft delete).
        
        Args:
            template_id: Template ID
            
        Returns:
            True if deleted, False if not found
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(TemplateModel).where(TemplateModel.id == template_id)
            result = await session.execute(query)
            template_model = result.scalar_one_or_none()
            
            if not template_model:
                return False
            
            template_model.is_active = False
            template_model.updated_at = datetime.utcnow()
            
            await session.flush()
            return True
    
    async def search_templates(
        self,
        query: str,
        stack: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Template]:
        """
        Search templates by query string.
        
        Args:
            query: Search query
            stack: Filter by stack
            tags: Filter by tags
            limit: Maximum results
            
        Returns:
            List of matching templates
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            # Build search query
            search_query = select(TemplateModel).options(
                selectinload(TemplateModel.stack)
            ).where(TemplateModel.is_active == True)
            
            # Add text search
            search_terms = query.lower().split()
            search_conditions = []
            
            for term in search_terms:
                search_conditions.append(
                    or_(
                        TemplateModel.name.ilike(f"%{term}%"),
                        TemplateModel.description.ilike(f"%{term}%"),
                        TemplateModel.technologies.contains([term]),
                        TemplateModel.tags.contains([term])
                    )
                )
            
            if search_conditions:
                search_query = search_query.where(and_(*search_conditions))
            
            # Apply filters
            if stack:
                search_query = search_query.join(Stack).where(Stack.name == stack)
            
            if tags:
                for tag in tags:
                    search_query = search_query.where(TemplateModel.tags.contains([tag]))
            
            # Order by relevance (quality score for now)
            search_query = search_query.order_by(desc(TemplateModel.quality_score))
            search_query = search_query.limit(limit)
            
            # Execute query
            result = await session.execute(search_query)
            template_models = result.scalars().all()
            
            # Convert to Pydantic models
            templates = []
            for template_model in template_models:
                template = self._model_to_pydantic(template_model)
                templates.append(template)
            
            return templates
    
    async def download_template(
        self, 
        template_id: str, 
        format: str = "zip",
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Download a template as an archive.
        
        Args:
            template_id: Template ID
            format: Archive format (zip, tar.gz)
            user_id: User ID (if authenticated)
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Archive bytes if successful, None otherwise
        """
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            # Get template
            query = select(TemplateModel).where(TemplateModel.id == template_id)
            result = await session.execute(query)
            template_model = result.scalar_one_or_none()
            
            if not template_model or not template_model.path:
                return None
            
            # Record download
            download = TemplateDownload(
                template_id=template_model.id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                format=format
            )
            session.add(download)
            await session.flush()
            
            # Create archive
            template_path = Path(template_model.path)
            if not template_path.exists():
                logger.warning(f"Template path does not exist: {template_path}")
                return None
            
            # Create temporary archive
            with tempfile.NamedTemporaryFile(suffix=f".{format}") as temp_file:
                if format == "zip":
                    with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(template_path):
                            for file in files:
                                file_path = Path(root) / file
                                arc_path = file_path.relative_to(template_path)
                                zipf.write(file_path, arc_path)
                
                # Read archive bytes
                temp_file.seek(0)
                archive_bytes = temp_file.read()
                
                return archive_bytes
    
    async def sync_from_github(
        self, 
        github_url: str, 
        stack: str = "other"
    ) -> Optional[Template]:
        """
        Sync a template from GitHub repository.
        
        Args:
            github_url: GitHub repository URL
            stack: Target stack
            
        Returns:
            Created/updated template
        """
        try:
            # Parse GitHub URL
            parsed_url = urlparse(github_url)
            if "github.com" not in parsed_url.netloc:
                raise ValueError("Not a GitHub URL")
            
            # Extract owner and repo
            path_parts = parsed_url.path.strip("/").split("/")
            if len(path_parts) < 2:
                raise ValueError("Invalid GitHub URL format")
            
            owner, repo = path_parts[0], path_parts[1]
            
            # Get repository info from GitHub API
            async with aiohttp.ClientSession() as session:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                async with session.get(api_url) as response:
                    if response.status != 200:
                        logger.error(f"GitHub API error: {response.status}")
                        return None
                    
                    repo_data = await response.json()
            
            # Create template data
            template_data = {
                "name": repo,
                "stack": stack,
                "description": repo_data.get("description", ""),
                "upstream_url": github_url,
                "author": repo_data.get("owner", {}).get("login", ""),
                "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None,
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "technologies": self._extract_technologies(repo_data),
                "tags": self._extract_tags(repo_data),
                "quality_score": self._calculate_quality_score(repo_data)
            }
            
            # Check if template already exists
            existing_template = await self.get_template(repo, stack)
            if existing_template:
                # Update existing template
                return await self.update_template(existing_template.id, template_data)
            else:
                # Create new template
                return await self.create_template(template_data)
                
        except Exception as e:
            logger.error(f"Error syncing from GitHub: {e}")
            return None
    
    def _model_to_pydantic(self, template_model: TemplateModel) -> Template:
        """Convert SQLAlchemy model to Pydantic model."""
        return Template(
            name=template_model.name,
            description=template_model.description or "No description available",
            stack=template_model.stack.name if template_model.stack else "other",
            path=template_model.path or "/templates/default",
            upstream_url=template_model.upstream_url,
            version=template_model.version,
            author=template_model.author or "Unknown",
            license=template_model.license or "MIT",
            tags=template_model.tags or [],
            technologies=template_model.technologies or [],
            features=template_model.features or [],
            dependencies=template_model.dependencies or {},
            min_python_version=template_model.min_python_version,
            min_node_version=template_model.min_node_version,
            stars=template_model.stars,
            forks=template_model.forks,
            growth_rate=template_model.growth_rate,
            quality_score=template_model.quality_score,
            created_at=template_model.created_at,
            updated_at=template_model.updated_at
        )
    
    def _extract_technologies(self, repo_data: Dict[str, Any]) -> List[str]:
        """Extract technologies from GitHub repository data."""
        technologies = []
        
        # Check topics
        topics = repo_data.get("topics", [])
        technologies.extend(topics)
        
        # Check language
        language = repo_data.get("language")
        if language:
            technologies.append(language.lower())
        
        return list(set(technologies))
    
    def _extract_tags(self, repo_data: Dict[str, Any]) -> List[str]:
        """Extract tags from GitHub repository data."""
        tags = []
        
        # Add topics as tags
        topics = repo_data.get("topics", [])
        tags.extend(topics)
        
        # Add language as tag
        language = repo_data.get("language")
        if language:
            tags.append(language.lower())
        
        # Add common tags based on repository characteristics
        if repo_data.get("has_issues"):
            tags.append("has-issues")
        if repo_data.get("has_wiki"):
            tags.append("has-wiki")
        if repo_data.get("has_pages"):
            tags.append("has-pages")
        
        return list(set(tags))
    
    def _calculate_quality_score(self, repo_data: Dict[str, Any]) -> float:
        """Calculate quality score based on repository metrics."""
        score = 0.0
        
        # Stars (0-0.3)
        stars = repo_data.get("stargazers_count", 0)
        if stars > 0:
            score += min(0.3, stars / 1000)
        
        # Forks (0-0.2)
        forks = repo_data.get("forks_count", 0)
        if forks > 0:
            score += min(0.2, forks / 100)
        
        # Has documentation (0-0.2)
        if repo_data.get("has_wiki") or repo_data.get("has_pages"):
            score += 0.2
        
        # Has issues enabled (0-0.1)
        if repo_data.get("has_issues"):
            score += 0.1
        
        # Has description (0-0.1)
        if repo_data.get("description"):
            score += 0.1
        
        # Has license (0-0.1)
        if repo_data.get("license"):
            score += 0.1
        
        return min(1.0, score)


# Global service instance
template_service = TemplateService()
