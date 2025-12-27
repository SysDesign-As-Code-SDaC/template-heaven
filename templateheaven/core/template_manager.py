"""
Template Manager for Template Heaven.

This module provides the core template management functionality including
discovery, filtering, and retrieval of templates from bundled data.
"""

import yaml
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache

from .models import Template, StackCategory, TemplateSearchResult
from .stack_config import get_stack_config_manager
from .github_search import GitHubSearchService
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
        self.stack_config = get_stack_config_manager()
        self.github_search = GitHubSearchService(self.config)
        self.prefer_github = self.config.get('prefer_github', False) or bool(self.config.get('github_token'))

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
    
    def list_templates(
        self,
        stack: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None
        , include_archived: bool = False
        , use_github: Optional[bool] = None
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
        # Prefer GitHub live discovery if configured
        templates: List[Template] = []
        if use_github is None:
            use_github = self.prefer_github

        if use_github and self.github_search.github_available:
            try:
                if stack:
                    github_candidates = asyncio.run(self.github_search.discover_templates_for_stack(stack, limit=50))
                else:
                    # If no stack, fall back to local for now
                    github_candidates = []

                # Convert candidates to Template objects where possible
                for candidate in github_candidates:
                    # Candidates may be dicts or objects, handle both
                    if isinstance(candidate, dict):
                        repo_data = candidate.get('repository', {})
                        stack_suggestions = candidate.get('stack_suggestions', []) or candidate.get('repository', {}).get('topics', [])
                    else:
                        repo_data = getattr(candidate, 'repository', {}) or {}
                        # Candidate objects may have a 'repository' attr with topics
                        stack_suggestions = getattr(candidate, 'stack_suggestions', None) or getattr(repo_data, 'get', lambda k, d: d)('topics', [])
                    # Attempt to infer stack from candidate
                    # Some github search candidates include stack suggestions
                    stack_enum = None
                    if stack_suggestions:
                        try:
                            stack_enum = self._infer_stack_enum(stack_suggestions)
                        except Exception:
                            stack_enum = StackCategory.BACKEND
                    else:
                        stack_enum = StackCategory.BACKEND

                    template = Template(
                        name=repo_data.get('name', ''),
                        stack=stack_enum,
                        description=repo_data.get('description', ''),
                        path=f"github:{repo_data.get('full_name', '')}",
                        tags=repo_data.get('topics', []) or [],
                        dependencies={}
                    )
                    # set archived if repository metadata contains an archived flag
                    if repo_data.get('archived'):
                        template.archived = True
                    templates.append(template)

                # If no discoveries, fallback to bundled templates
                if len(templates) == 0:
                    templates = self.bundled_templates.copy()
            except Exception:
                templates = self.bundled_templates.copy()
        else:
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
        
        # Filter out archived templates unless explicitly asked to include them
        if not include_archived:
            templates = [t for t in templates if not getattr(t, 'archived', False)]

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
            'template_count': len([t for t in self.bundled_templates if t.stack == stack and not getattr(t, 'archived', False)]),
            'templates': [t.name for t in self.bundled_templates if t.stack == stack and not getattr(t, 'archived', False)]
        }
    
    def search_templates(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.1
        , use_github: Optional[bool] = None
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
        # Determine whether to use GitHub search
        if use_github is None:
            use_github = self.prefer_github
        results = []
        
        if use_github and self.github_search.github_available:
            try:
                github_results = asyncio.run(self.github_search.search_github_templates(query, stack=None, min_stars=50, limit=limit))
                # Filter by min_score and return
                return [r for r in github_results if r.score >= min_score][:limit]
            except Exception:
                # fall back to local search
                pass

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

    def _infer_stack_enum(self, suggestions: List[str]) -> StackCategory:
        """
        Infer a StackCategory from a list of suggestion strings.
        """
        if not suggestions:
            return StackCategory.BACKEND

        # Common topic to stack category mapping for better inference
        topic_map = {
            'nextjs': StackCategory.FULLSTACK,
            'next.js': StackCategory.FULLSTACK,
            'next': StackCategory.FULLSTACK,
            'react': StackCategory.FRONTEND,
            'vite': StackCategory.FRONTEND,
            'vue': StackCategory.FRONTEND,
            'angular': StackCategory.FRONTEND,
            'fastapi': StackCategory.BACKEND,
            'flask': StackCategory.BACKEND,
            'django': StackCategory.BACKEND,
            'python': StackCategory.BACKEND,
            'nodejs': StackCategory.BACKEND,
            'typescript': StackCategory.FRONTEND,
            'pytorch': StackCategory.AI_ML,
            'tensorflow': StackCategory.AI_ML,
            'mlops': StackCategory.AI_ML,
            'terraform': StackCategory.DEVOPS,
            'docker': StackCategory.DEVOPS,
            'kubernetes': StackCategory.DEVOPS,
            'trpc': StackCategory.FULLSTACK,
            'prisma': StackCategory.FULLSTACK,
            'gold-standard': StackCategory.GOLD_STANDARD,
        }
        # Normalize suggestions and attempt direct mapping, then topic mapping
        for s in suggestions:
            s_lower = s.lower()
            try:
                return StackCategory(s_lower)
            except Exception:
                pass

            # Try mapping via known topic map
            if s_lower in topic_map:
                return topic_map[s_lower]

        return StackCategory.BACKEND
        return StackCategory.BACKEND
    
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
            # Basic validation
            if not template.name or not template.description:
                return False

            # Check name format
            if len(template.name) > 50 or not template.name.replace('-', '').replace('_', '').isalnum():
                return False

            # Check tags
            if len(template.tags) > 10:
                return False

            # Stack-specific validation
            stack_config = self.stack_config.get_stack_config(template.stack.value)
            if stack_config:
                # Check if template meets stack requirements
                template_data = {
                    'stars': getattr(template, 'stars', 0),
                    'forks': getattr(template, 'forks', 0),
                    'growth_rate': getattr(template, 'growth_rate', 0.0),
                    'technologies': getattr(template, 'technologies', [])
                }

                validation_result = self.stack_config.validate_template_for_stack(
                    template_data, template.stack.value
                )

                if not validation_result.get('valid', True):
                    logger.warning(f"Template {template.name} failed stack validation: {validation_result.get('issues', [])}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False

    def validate_template_for_stack(self, template: Template, stack_name: str) -> Dict[str, Any]:
        """
        Comprehensive validation with stack-specific requirements.

        Args:
            template: Template to validate
            stack_name: Stack to validate against

        Returns:
            Detailed validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'quality_score': 0.0,
            'stack_requirements': {}
        }

        try:
            # Basic validation
            if not template.name:
                results['issues'].append("Template name is required")
                results['valid'] = False

            if not template.description:
                results['issues'].append("Template description is required")
                results['valid'] = False

            if len(template.name) > 50:
                results['issues'].append("Template name too long (max 50 characters)")
                results['valid'] = False

            if len(template.tags) > 10:
                results['issues'].append("Too many tags (max 10)")
                results['valid'] = False

            # Stack-specific validation
            stack_config = self.stack_config.get_stack_config(stack_name)
            if stack_config:
                results['stack_requirements'] = {
                    'quality_standards': stack_config.quality_standards,
                    'min_stars': stack_config.requirements.min_stars,
                    'min_forks': stack_config.requirements.min_forks,
                    'min_growth_rate': stack_config.requirements.min_growth_rate
                }

                # Check requirements
                template_stars = getattr(template, 'stars', 0)
                template_forks = getattr(template, 'forks', 0)
                template_growth = getattr(template, 'growth_rate', 0.0)

                if template_stars < stack_config.requirements.min_stars:
                    results['issues'].append(
                        f"Insufficient stars: {template_stars} < {stack_config.requirements.min_stars}"
                    )

                if template_forks < stack_config.requirements.min_forks:
                    results['issues'].append(
                        f"Insufficient forks: {template_forks} < {stack_config.requirements.min_forks}"
                    )

                if template_growth < stack_config.requirements.min_growth_rate:
                    results['warnings'].append(
                        f"Low growth rate: {template_growth:.1%} < {stack_config.requirements.min_growth_rate:.1%}"
                    )

                # Check technology alignment
                template_tech = set(getattr(template, 'technologies', []))
                stack_tech = set(stack_config.technologies)

                if template_tech and stack_tech:
                    overlap = template_tech.intersection(stack_tech)
                    if not overlap:
                        results['warnings'].append(
                            f"No technology overlap with stack: {template_tech} vs {stack_tech}"
                        )
                    elif len(overlap) / len(template_tech) < 0.5:
                        results['recommendations'].append(
                            f"Low technology alignment ({len(overlap)}/{len(template_tech)} technologies match)"
                        )

                # Calculate quality score
                quality_score = self._calculate_quality_score(template, stack_config)
                results['quality_score'] = quality_score

                if quality_score < 7.0:
                    results['warnings'].append(f"Low quality score: {quality_score:.1f}/10")
            else:
                results['warnings'].append(f"No configuration found for stack: {stack_name}")

            # Update validity
            results['valid'] = len(results['issues']) == 0

        except Exception as e:
            logger.error(f"Stack validation failed for template {template.name}: {e}")
            results['issues'].append(f"Validation error: {str(e)}")
            results['valid'] = False

        return results

    def _calculate_quality_score(self, template: Template, stack_config) -> float:
        """Calculate quality score based on stack requirements."""
        score = 0.0
        max_score = 10.0

        # Stars contribution (2 points)
        stars_ratio = min(getattr(template, 'stars', 0) / stack_config.requirements.min_stars, 2.0)
        score += stars_ratio

        # Forks contribution (1 point)
        forks_ratio = min(getattr(template, 'forks', 0) / stack_config.requirements.min_forks, 1.0)
        score += forks_ratio

        # Growth rate contribution (1 point)
        growth_ratio = min(getattr(template, 'growth_rate', 0.0) / stack_config.requirements.min_growth_rate, 1.0)
        score += growth_ratio

        # Documentation quality (2 points) - estimated
        if len(template.description) > 50:
            score += 2.0
        elif len(template.description) > 20:
            score += 1.5
        else:
            score += 0.5

        # Technology alignment (2 points)
        template_tech = set(getattr(template, 'technologies', []))
        stack_tech = set(stack_config.technologies)
        if template_tech and stack_tech:
            overlap_ratio = len(template_tech.intersection(stack_tech)) / len(template_tech)
            score += 2.0 * overlap_ratio
        else:
            score += 1.0  # Neutral if no technology data

        # Tags quality (1 point)
        if 3 <= len(template.tags) <= 8:
            score += 1.0
        elif len(template.tags) > 0:
            score += 0.5

        # Normalize to 10-point scale
        return min(score, max_score)

    async def search_all_templates(
        self,
        query: str,
        stack: Optional[str] = None,
        include_github: bool = True,
        github_limit: int = 10,
        total_limit: int = 20
    ) -> List[TemplateSearchResult]:
        """
        Comprehensive search across local templates and GitHub.

        Args:
            query: Search query
            stack: Optional stack filter
            include_github: Whether to include GitHub search results
            github_limit: Maximum GitHub results to include
            total_limit: Maximum total results to return

        Returns:
            Combined search results from local and GitHub
        """
        all_results = []

        # Search local templates
        local_results = self.search_templates(query, limit=total_limit)
        all_results.extend(local_results)

        # Search GitHub if requested
        if include_github:
            try:
                # Check cache first for search results
                cache_key = f"github_search_{query}_{stack or 'all'}"
                cached_results = self.cache.get_cached_search_results(query, stack)

                if cached_results:
                    logger.debug(f"Using cached GitHub search results for: {query}")
                    github_results = cached_results
                else:
                    # Perform live search
                    github_results = await self.github_search.search_github_templates(
                        query=query,
                        stack=stack,
                        limit=github_limit
                    )

                    # Cache the results for future use
                    if github_results:
                        self.cache.cache_search_results(query, stack, github_results, ttl_seconds=3600)

                # Mark GitHub results as external
                for result in github_results:
                    result.match_reason = f"GitHub: {result.match_reason}"

                all_results.extend(github_results)

            except ImportError as e:
                logger.info(f"GitHub search not available: {e}")
            except Exception as e:
                logger.warning(f"GitHub search failed: {e}")

        # Sort all results by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Limit total results
        return all_results[:total_limit]

    async def discover_templates_for_stack(
        self,
        stack_name: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Discover new template candidates for a specific stack.

        Args:
            stack_name: Stack to find templates for
            limit: Maximum candidates to return

        Returns:
            List of template candidates with analysis
        """
        return await self.github_search.discover_templates_for_stack(stack_name, limit)

    async def analyze_github_repository(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a GitHub repository for template potential.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Analysis results or None if failed
        """
        return await self.github_search.analyze_repository(repo_url)

    async def get_github_rate_limit(self) -> Dict[str, Any]:
        """
        Get GitHub API rate limit status.

        Returns:
            Rate limit information
        """
        return await self.github_search.get_rate_limit_status()

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
