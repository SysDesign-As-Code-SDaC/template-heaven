"""
Tests for Template Manager.

This module contains unit tests for the TemplateManager class including
template discovery, filtering, and search functionality.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from templateheaven.core.template_manager import TemplateManager
from templateheaven.core.models import Template, StackCategory
from templateheaven.config.settings import Config


class TestTemplateManager:
    """Test TemplateManager class."""
    
    def test_initialization(self, mock_config, sample_stacks_data):
        """Test template manager initialization."""
        with patch('templateheaven.core.template_manager.Path') as mock_path:
            mock_path.return_value.parent.parent = Path("/test")
            mock_path.return_value.parent.parent.__truediv__ = Mock(return_value=Path("/test/data"))
            mock_path.return_value.parent.parent.__truediv__.return_value.__truediv__ = Mock(return_value=Path("/test/data/stacks.yaml"))
            mock_path.return_value.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=yaml.dump(sample_stacks_data))):
                with patch('yaml.safe_load', return_value=sample_stacks_data):
                    manager = TemplateManager(mock_config)
                    
                    assert len(manager.bundled_templates) == 1
                    assert manager.bundled_templates[0].name == "react-vite"
    
    def test_initialization_missing_file(self, mock_config):
        """Test initialization with missing data file."""
        with patch('templateheaven.core.template_manager.Path') as mock_path:
            mock_path.return_value.parent.parent = Path("/test")
            mock_path.return_value.parent.parent.__truediv__ = Mock(return_value=Path("/test/data"))
            mock_path.return_value.parent.parent.__truediv__.return_value.__truediv__ = Mock(return_value=Path("/test/data/stacks.yaml"))
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(FileNotFoundError):
                TemplateManager(mock_config)
    
    def test_parse_templates(self, mock_config, sample_stacks_data):
        """Test parsing templates from stacks data."""
        with patch('templateheaven.core.template_manager.Path') as mock_path:
            mock_path.return_value.parent.parent = Path("/test")
            mock_path.return_value.parent.parent.__truediv__ = Mock(return_value=Path("/test/data"))
            mock_path.return_value.parent.parent.__truediv__.return_value.__truediv__ = Mock(return_value=Path("/test/data/stacks.yaml"))
            mock_path.return_value.exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data=yaml.dump(sample_stacks_data))):
                with patch('yaml.safe_load', return_value=sample_stacks_data):
                    manager = TemplateManager(mock_config)
                    
                    templates = manager._parse_templates()
                    
                    assert len(templates) == 1
                    assert templates[0].name == "react-vite"
                    assert templates[0].stack == StackCategory.FRONTEND
                    assert templates[0].path == "bundled/frontend/react-vite"
    
    def test_list_templates_no_filters(self, mock_template_manager):
        """Test listing templates without filters."""
        templates = mock_template_manager.list_templates()
        
        assert len(templates) > 0
        assert all(isinstance(t, Template) for t in templates)
    
    def test_list_templates_with_stack_filter(self, mock_template_manager):
        """Test listing templates with stack filter."""
        templates = mock_template_manager.list_templates(stack="frontend")
        
        assert len(templates) > 0
        assert all(t.stack == StackCategory.FRONTEND for t in templates)
    
    def test_list_templates_with_invalid_stack(self, mock_template_manager):
        """Test listing templates with invalid stack filter."""
        templates = mock_template_manager.list_templates(stack="invalid-stack")
        
        assert len(templates) == 0
    
    def test_list_templates_with_tags_filter(self, mock_template_manager):
        """Test listing templates with tags filter."""
        templates = mock_template_manager.list_templates(tags=["react"])
        
        assert len(templates) > 0
        assert all(any(t.has_tag("react") for t in templates))
    
    def test_list_templates_with_search_filter(self, mock_template_manager):
        """Test listing templates with search filter."""
        templates = mock_template_manager.list_templates(search="react")
        
        assert len(templates) > 0
        assert all(t.matches_search("react") for t in templates)
    
    def test_list_templates_multiple_filters(self, mock_template_manager):
        """Test listing templates with multiple filters."""
        templates = mock_template_manager.list_templates(
            stack="frontend",
            tags=["react"],
            search="typescript"
        )
        
        assert len(templates) >= 0
        for template in templates:
            assert template.stack == StackCategory.FRONTEND
            assert template.has_tag("react")
            assert template.matches_search("typescript")
    
    def test_get_template_existing(self, mock_template_manager):
        """Test getting an existing template."""
        # First get a template name from the list
        templates = mock_template_manager.list_templates()
        if templates:
            template_name = templates[0].name
            template = mock_template_manager.get_template(template_name)
            
            assert template is not None
            assert template.name == template_name
    
    def test_get_template_nonexistent(self, mock_template_manager):
        """Test getting a non-existent template."""
        template = mock_template_manager.get_template("nonexistent-template")
        
        assert template is None
    
    def test_get_stacks(self, mock_template_manager):
        """Test getting all available stacks."""
        stacks = mock_template_manager.get_stacks()
        
        assert len(stacks) > 0
        assert all(isinstance(stack, StackCategory) for stack in stacks)
        assert StackCategory.FRONTEND in stacks
        assert StackCategory.BACKEND in stacks
    
    def test_get_stack_info(self, mock_template_manager):
        """Test getting stack information."""
        stack_info = mock_template_manager.get_stack_info(StackCategory.FRONTEND)
        
        assert "name" in stack_info
        assert "description" in stack_info
        assert "template_count" in stack_info
        assert "templates" in stack_info
        assert isinstance(stack_info["template_count"], int)
        assert isinstance(stack_info["templates"], list)
    
    def test_search_templates(self, mock_template_manager):
        """Test searching templates."""
        results = mock_template_manager.search_templates("react")
        
        assert len(results) >= 0
        assert all(0.0 <= result.score <= 1.0 for result in results)
        assert all(result.template.matches_search("react") for result in results)
    
    def test_search_templates_with_limit(self, mock_template_manager):
        """Test searching templates with limit."""
        results = mock_template_manager.search_templates("react", limit=5)
        
        assert len(results) <= 5
    
    def test_search_templates_with_min_score(self, mock_template_manager):
        """Test searching templates with minimum score."""
        results = mock_template_manager.search_templates("react", min_score=0.5)
        
        assert all(result.score >= 0.5 for result in results)
    
    def test_calculate_relevance_score_exact_name_match(self, mock_template_manager):
        """Test relevance score calculation for exact name match."""
        template = Template(
            name="react-template",
            stack=StackCategory.FRONTEND,
            description="A React template",
            path="test",
            tags=["react"]
        )
        
        score = mock_template_manager._calculate_relevance_score(template, "react-template")
        assert score == 1.0
    
    def test_calculate_relevance_score_name_contains(self, mock_template_manager):
        """Test relevance score calculation for name contains match."""
        template = Template(
            name="react-vite-template",
            stack=StackCategory.FRONTEND,
            description="A React template",
            path="test",
            tags=["react"]
        )
        
        score = mock_template_manager._calculate_relevance_score(template, "react")
        assert score == 0.8
    
    def test_calculate_relevance_score_description_match(self, mock_template_manager):
        """Test relevance score calculation for description match."""
        template = Template(
            name="template",
            stack=StackCategory.FRONTEND,
            description="A React application template",
            path="test",
            tags=["react"]
        )
        
        score = mock_template_manager._calculate_relevance_score(template, "react")
        assert score == 0.6
    
    def test_calculate_relevance_score_tag_match(self, mock_template_manager):
        """Test relevance score calculation for tag match."""
        template = Template(
            name="template",
            stack=StackCategory.FRONTEND,
            description="A template",
            path="test",
            tags=["react", "typescript", "vite"]
        )
        
        score = mock_template_manager._calculate_relevance_score(template, "react")
        assert score == 0.4  # 0.4 * (1/3) = 0.133, but normalized
    
    def test_calculate_relevance_score_stack_match(self, mock_template_manager):
        """Test relevance score calculation for stack match."""
        template = Template(
            name="template",
            stack=StackCategory.FRONTEND,
            description="A template",
            path="test",
            tags=[]
        )
        
        score = mock_template_manager._calculate_relevance_score(template, "frontend")
        assert score == 0.3
    
    def test_get_match_reason_exact_name(self, mock_template_manager):
        """Test getting match reason for exact name match."""
        template = Template(
            name="react-template",
            stack=StackCategory.FRONTEND,
            description="A template",
            path="test",
            tags=[]
        )
        
        reason = mock_template_manager._get_match_reason(template, "react-template")
        assert reason == "Exact name match"
    
    def test_get_match_reason_name_contains(self, mock_template_manager):
        """Test getting match reason for name contains match."""
        template = Template(
            name="react-vite-template",
            stack=StackCategory.FRONTEND,
            description="A template",
            path="test",
            tags=[]
        )
        
        reason = mock_template_manager._get_match_reason(template, "react")
        assert reason == "Name contains query"
    
    def test_get_match_reason_description(self, mock_template_manager):
        """Test getting match reason for description match."""
        template = Template(
            name="template",
            stack=StackCategory.FRONTEND,
            description="A React template",
            path="test",
            tags=[]
        )
        
        reason = mock_template_manager._get_match_reason(template, "react")
        assert reason == "Description contains query"
    
    def test_get_match_reason_tag(self, mock_template_manager):
        """Test getting match reason for tag match."""
        template = Template(
            name="template",
            stack=StackCategory.FRONTEND,
            description="A template",
            path="test",
            tags=["react"]
        )
        
        reason = mock_template_manager._get_match_reason(template, "react")
        assert reason == "Tag match"
    
    def test_get_match_reason_stack(self, mock_template_manager):
        """Test getting match reason for stack match."""
        template = Template(
            name="template",
            stack=StackCategory.FRONTEND,
            description="A template",
            path="test",
            tags=[]
        )
        
        reason = mock_template_manager._get_match_reason(template, "frontend")
        assert reason == "Stack category match"
    
    def test_get_template_stats(self, mock_template_manager):
        """Test getting template statistics."""
        stats = mock_template_manager.get_template_stats()
        
        assert "total_templates" in stats
        assert "stacks" in stats
        assert "tags" in stats
        assert "dependencies" in stats
        
        assert isinstance(stats["total_templates"], int)
        assert isinstance(stats["stacks"], dict)
        assert isinstance(stats["tags"], dict)
        assert isinstance(stats["dependencies"], list)
    
    def test_validate_template_valid(self, mock_template_manager, sample_template):
        """Test validating a valid template."""
        is_valid = mock_template_manager.validate_template(sample_template)
        assert is_valid is True
    
    def test_validate_template_invalid_name(self, mock_template_manager):
        """Test validating a template with invalid name."""
        template = Template(
            name="",  # Empty name
            stack=StackCategory.FRONTEND,
            description="Test",
            path="test"
        )
        
        is_valid = mock_template_manager.validate_template(template)
        assert is_valid is False
    
    def test_validate_template_invalid_description(self, mock_template_manager):
        """Test validating a template with invalid description."""
        template = Template(
            name="test",
            stack=StackCategory.FRONTEND,
            description="",  # Empty description
            path="test"
        )
        
        is_valid = mock_template_manager.validate_template(template)
        assert is_valid is False
    
    def test_validate_template_too_many_tags(self, mock_template_manager):
        """Test validating a template with too many tags."""
        template = Template(
            name="test",
            stack=StackCategory.FRONTEND,
            description="Test",
            path="test",
            tags=["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11"]  # 11 tags
        )
        
        is_valid = mock_template_manager.validate_template(template)
        assert is_valid is False
    
    def test_get_popular_templates(self, mock_template_manager):
        """Test getting popular templates."""
        popular = mock_template_manager.get_popular_templates(limit=5)
        
        assert len(popular) <= 5
        assert all(isinstance(template, Template) for template in popular)
    
    def test_get_templates_by_dependency(self, mock_template_manager):
        """Test getting templates by dependency."""
        templates = mock_template_manager.get_templates_by_dependency("react")
        
        assert len(templates) >= 0
        assert all("react" in [dep.lower() for dep in t.dependencies.keys()] for t in templates)
    
    def test_refresh_cache(self, mock_template_manager):
        """Test refreshing template cache."""
        # This should not raise an exception
        mock_template_manager.refresh_cache()
    
    def test_get_cache_stats(self, mock_template_manager):
        """Test getting cache statistics."""
        stats = mock_template_manager.get_cache_stats()
        
        assert "total_entries" in stats
        assert "active_entries" in stats
        assert "expired_entries" in stats
        assert "total_size_bytes" in stats
