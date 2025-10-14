"""
Template Manager for Template Heaven.

This module provides the core template management functionality including
discovery, filtering, and retrieval of templates from bundled data.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache

from .models import Template, StackCategory, TemplateSearchResult
from ..config.settings import Config
from ..utils.logger import get_logger
from ..utils.cache import Cache

logger = get_logger(__name__)


class TemplateManager:
    """
    Manages template discovery, caching, and retrieval.
    
    MVP functionality:
    - Load bundled template metadata from YAML
    - Simple file-based caching
    - Template listing and filtering
    - Basic template retrieval
    - Search functionality
    
    Attributes:
        config: Configuration instance
        cache: Cache instance for template metadata
        bundled_templates: List of bundled templates
        stacks_data: Raw stacks data from YAML
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize template manager.
        
        Args:
            config: Configuration instance (creates default if None)
            
        Raises:
            FileNotFoundError: If bundled data files are missing
            yaml.YAMLError: If YAML data is invalid
        """
        self.config = config or Config()
        self.cache = Cache(
            cache_dir=self.config.get_metadata_cache_dir(),
            default_ttl=3600  # 1 hour
        )
        
        # Load bundled template data
        self._load_bundled_data()
        
        logger.info(f"TemplateManager initialized with {len(self.bundled_templates)} templates")
    
    def _load_bundled_data(self) -> None:
        """Load bundled template data from YAML file."""
        try:
            # Get path to bundled data
            data_dir = Path(__file__).parent.parent / "data"
            stacks_file = data_dir / "stacks.yaml"
            
            if not stacks_file.exists():
                raise FileNotFoundError(f"Bundled data file not found: {stacks_file}")
            
            # Load YAML data
            with open(stacks_file, 'r', encoding='utf-8') as f:
                self.stacks_data = yaml.safe_load(f)
            
            # Parse templates
            self.bundled_templates = self._parse_templates()
            
            logger.debug(f"Loaded {len(self.bundled_templates)} bundled templates")
            
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load bundled data: {e}")
            raise
    
    def _parse_templates(self) -> List[Template]:
        """
        Parse templates from stacks data.
        
        Returns:
            List of Template objects
        """
        templates = []
        
        for stack_name, stack_data in self.stacks_data.get('stacks', {}).items():
            try:
                stack_category = StackCategory(stack_name)
            except ValueError:
                logger.warning(f"Unknown stack category: {stack_name}")
                continue
            
            for template_data in stack_data.get('templates', []):
                try:
                    # Add stack information to template data
                    template_data['stack'] = stack_name
                    template_data['path'] = f"bundled/{stack_name}/{template_data['name']}"
                    
                    template = Template.from_dict(template_data)
                    templates.append(template)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse template {template_data.get('name', 'unknown')}: {e}")
                    continue
        
        return templates
    
    @lru_cache(maxsize=128)
    def list_templates(
        self,
        stack: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None
    ) -> List[Template]:
        """
        List available templates with optional filtering.
        
        Args:
            stack: Filter by stack category
            tags: Filter by tags (any match)
            search: Search in name, description, and tags
            
        Returns:
            List of matching templates
        """
        templates = self.bundled_templates.copy()
        
        # Filter by stack
        if stack:
            try:
                stack_category = StackCategory(stack)
                templates = [t for t in templates if t.stack == stack_category]
            except ValueError:
                logger.warning(f"Invalid stack category: {stack}")
                return []
        
        # Filter by tags
        if tags:
            filtered_templates = []
            for template in templates:
                if any(template.has_tag(tag) for tag in tags):
                    filtered_templates.append(template)
            templates = filtered_templates
        
        # Search filter
        if search:
            templates = [t for t in templates if t.matches_search(search)]
        
        logger.debug(f"Found {len(templates)} templates matching filters")
        return templates
    
    def get_template(self, name: str) -> Optional[Template]:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template instance or None if not found
        """
        for template in self.bundled_templates:
            if template.name == name:
                logger.debug(f"Found template: {name}")
                return template
        
        logger.debug(f"Template not found: {name}")
        return None
    
    def get_stacks(self) -> List[StackCategory]:
        """
        Get all available stack categories.
        
        Returns:
            List of stack categories
        """
        return list(StackCategory)
    
    def get_stack_info(self, stack: StackCategory) -> Dict[str, Any]:
        """
        Get information about a specific stack.
        
        Args:
            stack: Stack category
            
        Returns:
            Dictionary with stack information
        """
        stack_name = stack.value
        stack_data = self.stacks_data.get('stacks', {}).get(stack_name, {})
        
        return {
            'name': stack_data.get('name', StackCategory.get_display_name(stack)),
            'description': stack_data.get('description', StackCategory.get_description(stack)),
            'template_count': len([t for t in self.bundled_templates if t.stack == stack]),
            'templates': [t.name for t in self.bundled_templates if t.stack == stack]
        }
    
    def search_templates(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.1
    ) -> List[TemplateSearchResult]:
        """
        Search templates with relevance scoring.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum relevance score
            
        Returns:
            List of search results with scores
        """
        query_lower = query.lower()
        results = []
        
        for template in self.bundled_templates:
            score = self._calculate_relevance_score(template, query_lower)
            
            if score >= min_score:
                match_reason = self._get_match_reason(template, query_lower)
                results.append(TemplateSearchResult(
                    template=template,
                    score=score,
                    match_reason=match_reason
                ))
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _calculate_relevance_score(self, template: Template, query: str) -> float:
        """
        Calculate relevance score for a template against a query.
        
        Args:
            template: Template to score
            query: Search query (lowercase)
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        
        # Exact name match (highest score)
        if query == template.name.lower():
            score += 1.0
        elif query in template.name.lower():
            score += 0.8
        
        # Description match
        if query in template.description.lower():
            score += 0.6
        
        # Tag matches
        tag_matches = sum(1 for tag in template.tags if query in tag.lower())
        if tag_matches > 0:
            score += 0.4 * (tag_matches / len(template.tags))
        
        # Stack match
        if query in template.stack.value.lower():
            score += 0.3
        
        # Normalize score to 0.0-1.0 range
        return min(score, 1.0)
    
    def _get_match_reason(self, template: Template, query: str) -> str:
        """
        Get reason for template match.
        
        Args:
            template: Template that matched
            query: Search query (lowercase)
            
        Returns:
            Human-readable match reason
        """
        if query == template.name.lower():
            return "Exact name match"
        elif query in template.name.lower():
            return "Name contains query"
        elif query in template.description.lower():
            return "Description contains query"
        elif any(query in tag.lower() for tag in template.tags):
            return "Tag match"
        elif query in template.stack.value.lower():
            return "Stack category match"
        else:
            return "Partial match"
    
    def get_template_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available templates.
        
        Returns:
            Dictionary with template statistics
        """
        stats = {
            'total_templates': len(self.bundled_templates),
            'stacks': {},
            'tags': {},
            'dependencies': set()
        }
        
        # Count by stack
        for template in self.bundled_templates:
            stack_name = template.stack.value
            if stack_name not in stats['stacks']:
                stats['stacks'][stack_name] = 0
            stats['stacks'][stack_name] += 1
            
            # Count tags
            for tag in template.tags:
                if tag not in stats['tags']:
                    stats['tags'][tag] = 0
                stats['tags'][tag] += 1
            
            # Collect dependencies
            stats['dependencies'].update(template.dependencies.keys())
        
        # Convert set to list for JSON serialization
        stats['dependencies'] = list(stats['dependencies'])
        
        return stats
    
    def validate_template(self, template: Template) -> bool:
        """
        Validate a template.
        
        Args:
            template: Template to validate
            
        Returns:
            True if template is valid
        """
        try:
            # Check required fields
            if not template.name or not template.description:
                return False
            
            # Check name format
            if len(template.name) > 50 or not template.name.replace('-', '').replace('_', '').isalnum():
                return False
            
            # Check tags
            if len(template.tags) > 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False
    
    def get_popular_templates(self, limit: int = 10) -> List[Template]:
        """
        Get popular templates (based on common tags and features).
        
        Args:
            limit: Maximum number of templates to return
            
        Returns:
            List of popular templates
        """
        # Simple popularity scoring based on common tags
        popular_tags = ['typescript', 'react', 'python', 'api', 'docker', 'testing']
        
        scored_templates = []
        for template in self.bundled_templates:
            score = sum(1 for tag in template.tags if tag in popular_tags)
            scored_templates.append((template, score))
        
        # Sort by score and return top templates
        scored_templates.sort(key=lambda x: x[1], reverse=True)
        return [template for template, _ in scored_templates[:limit]]
    
    def get_templates_by_dependency(self, dependency: str) -> List[Template]:
        """
        Get templates that use a specific dependency.
        
        Args:
            dependency: Dependency name to search for
            
        Returns:
            List of templates using the dependency
        """
        matching_templates = []
        
        for template in self.bundled_templates:
            if dependency.lower() in [dep.lower() for dep in template.dependencies.keys()]:
                matching_templates.append(template)
        
        return matching_templates
    
    def refresh_cache(self) -> None:
        """Refresh template cache."""
        # Clear cache and reload data
        self.cache.clear()
        self._load_bundled_data()
        logger.info("Template cache refreshed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
