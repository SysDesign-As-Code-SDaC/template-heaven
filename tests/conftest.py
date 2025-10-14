"""
Pytest configuration and fixtures for Template Heaven tests.

This module provides common fixtures and configuration for the test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

from templateheaven.config.settings import Config
from templateheaven.core.template_manager import TemplateManager
from templateheaven.core.models import Template, ProjectConfig, StackCategory


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration for tests."""
    config = Config(config_dir=temp_dir / "config")
    return config


@pytest.fixture
def mock_template_manager(mock_config):
    """Create a mock template manager for tests."""
    return TemplateManager(mock_config)


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return Template(
        name="test-template",
        stack=StackCategory.FRONTEND,
        description="A test template for frontend development",
        path="bundled/frontend/test-template",
        tags=["react", "typescript", "vite"],
        dependencies={
            "react": "^18.2.0",
            "typescript": "^5.0.0",
            "vite": "^4.4.0"
        },
        upstream_url="https://github.com/test/test-template",
        version="1.0.0",
        author="Test Author",
        license="MIT",
        features=["TypeScript", "Hot Reload", "ESLint"],
        min_node_version="16.0.0"
    )


@pytest.fixture
def sample_project_config(sample_template):
    """Create a sample project configuration for testing."""
    return ProjectConfig(
        name="test-project",
        directory="/tmp",
        template=sample_template,
        author="Test User",
        license="MIT",
        package_manager="npm",
        description="A test project",
        version="0.1.0"
    )


@pytest.fixture
def sample_stacks_data():
    """Sample stacks data for testing."""
    return {
        "stacks": {
            "frontend": {
                "name": "Frontend Frameworks",
                "description": "Frontend frameworks and UI libraries",
                "templates": [
                    {
                        "name": "react-vite",
                        "description": "React + Vite + TypeScript starter",
                        "tags": ["react", "vite", "typescript"],
                        "dependencies": {
                            "react": "^18.2.0",
                            "typescript": "^5.0.0"
                        },
                        "upstream_url": "https://github.com/vitejs/vite",
                        "version": "1.0.0",
                        "author": "Vite Team",
                        "license": "MIT",
                        "features": ["TypeScript", "Hot Reload"],
                        "min_node_version": "16.0.0"
                    }
                ]
            }
        },
        "config": {
            "defaults": {
                "author": "Test User",
                "license": "MIT"
            }
        }
    }


@pytest.fixture
def mock_file_operations():
    """Create a mock file operations object."""
    mock_ops = Mock()
    mock_ops.copy_file.return_value = True
    mock_ops.copy_directory.return_value = True
    mock_ops.create_directory.return_value = True
    mock_ops.write_file.return_value = True
    mock_ops.read_file.return_value = "test content"
    return mock_ops


@pytest.fixture
def mock_cache():
    """Create a mock cache object."""
    mock_cache = Mock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.clear.return_value = True
    mock_cache.get_stats.return_value = {
        "total_entries": 0,
        "active_entries": 0,
        "expired_entries": 0,
        "total_size_bytes": 0
    }
    return mock_cache


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    mock_logger = Mock()
    mock_logger.debug = Mock()
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    mock_logger.error = Mock()
    return mock_logger


# Test data fixtures
@pytest.fixture
def test_templates():
    """Create a list of test templates."""
    return [
        Template(
            name="react-template",
            stack=StackCategory.FRONTEND,
            description="React template",
            path="bundled/frontend/react-template",
            tags=["react", "javascript"],
            dependencies={"react": "^18.0.0"}
        ),
        Template(
            name="vue-template",
            stack=StackCategory.FRONTEND,
            description="Vue template",
            path="bundled/frontend/vue-template",
            tags=["vue", "javascript"],
            dependencies={"vue": "^3.0.0"}
        ),
        Template(
            name="fastapi-template",
            stack=StackCategory.BACKEND,
            description="FastAPI template",
            path="bundled/backend/fastapi-template",
            tags=["python", "fastapi"],
            dependencies={"fastapi": "^0.100.0"}
        )
    ]


@pytest.fixture
def test_search_results():
    """Create test search results."""
    from templateheaven.core.models import TemplateSearchResult
    
    template = Template(
        name="react-typescript",
        stack=StackCategory.FRONTEND,
        description="React with TypeScript",
        path="bundled/frontend/react-typescript",
        tags=["react", "typescript"],
        dependencies={"react": "^18.0.0", "typescript": "^5.0.0"}
    )
    
    return [
        TemplateSearchResult(
            template=template,
            score=0.9,
            match_reason="Exact name match"
        )
    ]


# Configuration fixtures
@pytest.fixture
def test_config_data():
    """Test configuration data."""
    return {
        "cache_dir": "~/.templateheaven/cache",
        "default_author": "Test User",
        "default_license": "MIT",
        "github_token": None,
        "auto_update": True,
        "log_level": "INFO"
    }


# CLI fixtures
@pytest.fixture
def mock_click_context():
    """Create a mock Click context."""
    context = Mock()
    context.obj = {
        'config': Mock(),
        'template_manager': Mock()
    }
    return context


@pytest.fixture
def mock_console():
    """Create a mock Rich console."""
    mock_console = Mock()
    mock_console.print = Mock()
    return mock_console


# Utility fixtures
@pytest.fixture
def sample_project_structure(temp_dir):
    """Create a sample project structure for testing."""
    project_dir = temp_dir / "test-project"
    project_dir.mkdir()
    
    # Create basic structure
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()
    (project_dir / "docs").mkdir()
    
    # Create sample files
    (project_dir / "README.md").write_text("# Test Project")
    (project_dir / "package.json").write_text('{"name": "test-project"}')
    
    return project_dir


@pytest.fixture
def sample_template_variables():
    """Sample template variables for testing."""
    return {
        "project_name": "test-project",
        "project_description": "A test project",
        "author": "Test User",
        "license": "MIT",
        "version": "0.1.0",
        "package_manager": "npm",
        "template_name": "react-vite",
        "template_stack": "frontend"
    }


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "external: mark test as requiring external services")


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def create_temp_template_file(temp_dir: Path, name: str, content: str) -> Path:
        """Create a temporary template file."""
        template_file = temp_dir / name
        template_file.write_text(content)
        return template_file
    
    @staticmethod
    def create_temp_yaml_file(temp_dir: Path, name: str, data: Dict[str, Any]) -> Path:
        """Create a temporary YAML file."""
        import yaml
        yaml_file = temp_dir / name
        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)
        return yaml_file


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils
