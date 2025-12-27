#!/usr/bin/env python3
"""
AI Coding Agent Setup Script

This script sets up the AI coding agent context and configuration
for optimal collaboration between human developers and AI assistants.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import platform


class AIContextSetup:
    """Setup AI coding agent context and configuration."""
    
    def __init__(self, project_root: Path):
        """Initialize the AI context setup.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.cursor_rules_dir = project_root / ".cursor" / "rules"
        self.ai_config_dir = project_root / ".ai"
        
    def setup_ai_context(self) -> bool:
        """Set up complete AI coding agent context.
        
        Returns:
            True if setup was successful
        """
        try:
            print("🤖 Setting up AI Coding Agent Context...")
            
            # Create necessary directories
            self._create_directories()
            
            # Setup Cursor AI rules
            self._setup_cursor_rules()
            
            # Setup AI configuration
            self._setup_ai_config()
            
            # Setup development tools
            self._setup_dev_tools()
            
            # Setup pre-commit hooks
            self._setup_pre_commit_hooks()
            
            # Generate project context
            self._generate_project_context()
            
            # Setup AI-friendly documentation
            self._setup_ai_documentation()
            
            print("✅ AI Coding Agent Context setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up AI context: {e}")
            return False
    
    def _create_directories(self) -> None:
        """Create necessary directories for AI context."""
        directories = [
            self.cursor_rules_dir,
            self.ai_config_dir,
            self.project_root / ".vscode",
            self.project_root / "docs" / "ai",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def _setup_cursor_rules(self) -> None:
        """Setup Cursor AI rules for optimal AI assistance."""
        rules = {
            "project-context.mdc": self._get_project_context_rule(),
            "coding-standards.mdc": self._get_coding_standards_rule(),
            "ai-workflow.mdc": self._get_ai_workflow_rule(),
            "testing-standards.mdc": self._get_testing_standards_rule(),
            "documentation-standards.mdc": self._get_documentation_standards_rule(),
        }
        
        for filename, content in rules.items():
            rule_path = self.cursor_rules_dir / filename
            rule_path.write_text(content, encoding='utf-8')
            print(f"📝 Created Cursor rule: {filename}")
    
    def _setup_ai_config(self) -> None:
        """Setup AI configuration files."""
        # AI configuration
        ai_config = {
            "version": "1.0.0",
            "project_type": "python_fastapi",
            "ai_assistants": {
                "cursor": {
                    "enabled": True,
                    "rules_directory": ".cursor/rules",
                    "context_files": [
                        "README.md",
                        "pyproject.toml",
                        "requirements.txt",
                        "docs/README.md"
                    ]
                },
                "github_copilot": {
                    "enabled": True,
                    "context_aware": True,
                    "test_generation": True,
                    "documentation_generation": True
                },
                "claude": {
                    "enabled": True,
                    "context_window": "large",
                    "code_analysis": True,
                    "architecture_review": True
                }
            },
            "development_workflow": {
                "auto_formatting": True,
                "auto_imports": True,
                "auto_docstrings": True,
                "auto_tests": True,
                "auto_documentation": True
            },
            "quality_gates": {
                "type_hints_required": True,
                "docstrings_required": True,
                "test_coverage_minimum": 80,
                "security_scanning": True,
                "performance_monitoring": True
            }
        }
        
        config_path = self.ai_config_dir / "config.json"
        config_path.write_text(json.dumps(ai_config, indent=2), encoding='utf-8')
        print(f"⚙️ Created AI config: {config_path}")
    
    def _setup_dev_tools(self) -> None:
        """Setup development tools for AI assistance."""
        # VS Code settings
        vscode_settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.linting.enabled": True,
            "python.linting.flake8Enabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests"],
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True,
                "source.fixAll": True
            },
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/.pytest_cache": True,
                "**/htmlcov": True,
                "**/.coverage": True
            }
        }
        
        vscode_dir = self.project_root / ".vscode"
        settings_path = vscode_dir / "settings.json"
        settings_path.write_text(json.dumps(vscode_settings, indent=2), encoding='utf-8')
        print(f"🔧 Created VS Code settings: {settings_path}")
    
    def _setup_pre_commit_hooks(self) -> None:
        """Setup pre-commit hooks for AI-friendly development."""
        pre_commit_config = {
            "repos": [
                {
                    "repo": "https://github.com/pre-commit/pre-commit-hooks",
                    "rev": "v4.4.0",
                    "hooks": [
                        {"id": "trailing-whitespace"},
                        {"id": "end-of-file-fixer"},
                        {"id": "check-yaml"},
                        {"id": "check-added-large-files"},
                        {"id": "check-merge-conflict"},
                    ]
                },
                {
                    "repo": "https://github.com/psf/black",
                    "rev": "23.11.0",
                    "hooks": [{"id": "black", "language_version": "python3"}]
                },
                {
                    "repo": "https://github.com/pycqa/isort",
                    "rev": "5.12.0",
                    "hooks": [{"id": "isort"}]
                },
                {
                    "repo": "https://github.com/pycqa/flake8",
                    "rev": "6.1.0",
                    "hooks": [{"id": "flake8"}]
                },
                {
                    "repo": "https://github.com/pre-commit/mirrors-mypy",
                    "rev": "v1.7.1",
                    "hooks": [
                        {
                            "id": "mypy",
                            "additional_dependencies": ["types-all"]
                        }
                    ]
                },
                {
                    "repo": "https://github.com/PyCQA/bandit",
                    "rev": "1.7.5",
                    "hooks": [{"id": "bandit", "args": ["-r", "app/"]}]
                }
            ]
        }
        
        pre_commit_path = self.project_root / ".pre-commit-config.yaml"
        import yaml
        pre_commit_path.write_text(yaml.dump(pre_commit_config, default_flow_style=False), encoding='utf-8')
        print(f"🪝 Created pre-commit config: {pre_commit_path}")
    
    def _generate_project_context(self) -> None:
        """Generate comprehensive project context for AI."""
        context = {
            "project_name": self.project_root.name,
            "project_type": "FastAPI Python Service",
            "architecture": "Clean Architecture with Dependency Injection",
            "database": "PostgreSQL with SQLAlchemy ORM",
            "authentication": "JWT with refresh tokens",
            "testing": "Pytest with 90%+ coverage",
            "documentation": "Sphinx with auto-generated API docs",
            "deployment": "Docker with Kubernetes",
            "monitoring": "Prometheus metrics with structured logging",
            "security": "Comprehensive security scanning and validation",
            "ai_integration": "Full AI coding agent support with Cursor rules",
            "key_files": [
                "app/main.py - FastAPI application entry point",
                "app/core/models.py - SQLAlchemy models",
                "app/core/security.py - Authentication and security",
                "app/api/v1/ - API endpoints",
                "app/core/services/ - Business logic services",
                "tests/ - Comprehensive test suite",
                "docs/ - Complete documentation",
                ".cursor/rules/ - AI coding agent rules"
            ],
            "development_workflow": [
                "1. Create feature branch",
                "2. Write tests first (TDD)",
                "3. Implement feature with AI assistance",
                "4. Run quality checks (linting, type checking, security)",
                "5. Update documentation",
                "6. Submit pull request",
                "7. Code review with AI assistance",
                "8. Merge and deploy"
            ],
            "ai_capabilities": [
                "Code generation with type hints and docstrings",
                "Test case generation with edge cases",
                "Documentation generation and updates",
                "Security vulnerability detection",
                "Performance optimization suggestions",
                "Architecture review and recommendations",
                "Refactoring assistance with confidence",
                "Debugging and error analysis"
            ]
        }
        
        context_path = self.ai_config_dir / "project_context.json"
        context_path.write_text(json.dumps(context, indent=2), encoding='utf-8')
        print(f"📋 Generated project context: {context_path}")
    
    def _setup_ai_documentation(self) -> None:
        """Setup AI-friendly documentation structure."""
        ai_docs_dir = self.project_root / "docs" / "ai"
        
        # AI Development Guide
        ai_guide = """# AI Development Guide

## 🤖 AI Coding Agent Integration

This project is fully configured for optimal AI-human collaboration.

### Available AI Assistants

#### Cursor AI
- **Rules**: Located in `.cursor/rules/`
- **Context**: Full project understanding
- **Capabilities**: Code generation, refactoring, debugging

#### GitHub Copilot
- **Integration**: VS Code extension
- **Context**: File and project awareness
- **Capabilities**: Code completion, test generation

#### Claude
- **Integration**: API access
- **Context**: Large context window
- **Capabilities**: Architecture review, complex problem solving

### AI Development Workflow

1. **Planning**: Use AI to break down requirements
2. **Design**: Get AI assistance with architecture decisions
3. **Implementation**: Generate code with AI assistance
4. **Testing**: Create comprehensive tests with AI
5. **Documentation**: Generate and update docs with AI
6. **Review**: Use AI for code review and optimization

### Best Practices

- **Clear Context**: Provide comprehensive project documentation
- **Consistent Patterns**: Follow established coding patterns
- **Validation**: Always validate AI-generated code
- **Testing**: Ensure AI-generated code is thoroughly tested
- **Documentation**: Keep documentation updated with AI assistance

### AI Configuration

See `.ai/config.json` for detailed AI configuration.
"""
        
        guide_path = ai_docs_dir / "development_guide.md"
        guide_path.write_text(ai_guide, encoding='utf-8')
        print(f"📚 Created AI development guide: {guide_path}")
    
    def _get_project_context_rule(self) -> str:
        """Get project context rule for Cursor AI."""
        return """# Project Context for AI Coding Agents

## 🏗️ Project Overview

This is a **production-ready FastAPI Python service** with comprehensive software engineering best practices.

### Architecture
- **Framework**: FastAPI with async support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT with refresh tokens
- **Testing**: Pytest with 90%+ coverage
- **Documentation**: Sphinx with auto-generated API docs
- **Deployment**: Docker with Kubernetes
- **Monitoring**: Prometheus metrics with structured logging

### Key Directories
- `app/` - Main application code
- `app/core/` - Core business logic and models
- `app/api/` - API endpoints and routes
- `tests/` - Comprehensive test suite
- `docs/` - Complete documentation
- `.cursor/rules/` - AI coding agent rules

### Development Standards
- **Type Hints**: Required for all functions and methods
- **Docstrings**: Google-style with examples
- **Testing**: Unit, integration, and e2e tests
- **Security**: Comprehensive security scanning
- **Performance**: Monitoring and optimization
- **Documentation**: Auto-generated and maintained

### AI Integration
- **Cursor AI**: Full project context with rules
- **GitHub Copilot**: Context-aware code completion
- **Claude**: Architecture review and complex problem solving
- **Automated**: Code generation, testing, documentation

Always follow these standards when generating or modifying code.
"""
    
    def _get_coding_standards_rule(self) -> str:
        """Get coding standards rule for Cursor AI."""
        return """# Coding Standards for AI Coding Agents

## 🎯 Code Quality Standards

### Type Hints
- **Required**: All functions and methods must have type hints
- **Style**: Use modern Python typing (Python 3.9+)
- **Examples**: `def process_user(user: User) -> UserResponse:`

### Docstrings
- **Style**: Google-style docstrings
- **Required**: All public functions, classes, and methods
- **Format**: Include description, args, returns, raises, examples

### Error Handling
- **Custom Exceptions**: Use project-specific exception classes
- **Logging**: Log errors with appropriate levels
- **User-Friendly**: Provide clear error messages

### Code Style
- **Formatter**: Black with line length 88
- **Imports**: isort with black profile
- **Linting**: flake8 with project-specific rules
- **Type Checking**: mypy with strict mode

### Security
- **Input Validation**: Use Pydantic models
- **Authentication**: JWT with proper validation
- **Authorization**: Role-based access control
- **Secrets**: Use environment variables

### Performance
- **Async**: Use async/await for I/O operations
- **Database**: Optimize queries with proper indexing
- **Caching**: Implement Redis caching where appropriate
- **Monitoring**: Add performance metrics

Always generate code that follows these standards.
"""
    
    def _get_ai_workflow_rule(self) -> str:
        """Get AI workflow rule for Cursor AI."""
        return """# AI Workflow for Development

## 🤖 AI-Assisted Development Process

### Code Generation
1. **Understand Requirements**: Analyze user request and context
2. **Generate Code**: Create code following project standards
3. **Add Type Hints**: Include comprehensive type annotations
4. **Add Docstrings**: Include Google-style documentation
5. **Add Tests**: Generate corresponding test cases
6. **Update Documentation**: Update relevant documentation

### Test Generation
1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test API endpoints and database interactions
3. **Edge Cases**: Include boundary conditions and error cases
4. **Fixtures**: Create reusable test fixtures
5. **Coverage**: Ensure comprehensive test coverage

### Documentation Generation
1. **API Docs**: Update OpenAPI/Swagger documentation
2. **Code Comments**: Add inline comments for complex logic
3. **README Updates**: Update project documentation
4. **Architecture Docs**: Update system design documentation

### Code Review
1. **Security**: Check for security vulnerabilities
2. **Performance**: Identify performance bottlenecks
3. **Best Practices**: Ensure adherence to standards
4. **Testing**: Verify test coverage and quality
5. **Documentation**: Check documentation completeness

### Refactoring
1. **Identify Issues**: Find code smells and improvements
2. **Plan Changes**: Design refactoring approach
3. **Implement Safely**: Make changes with confidence
4. **Update Tests**: Ensure tests still pass
5. **Update Docs**: Update documentation as needed

Always follow this workflow when assisting with development.
"""
    
    def _get_testing_standards_rule(self) -> str:
        """Get testing standards rule for Cursor AI."""
        return """# Testing Standards for AI Coding Agents

## 🧪 Comprehensive Testing Requirements

### Test Structure
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test API endpoints and database interactions
- **End-to-End Tests**: Test complete user workflows
- **Performance Tests**: Test system performance and benchmarks

### Test Quality
- **Coverage**: Minimum 90% code coverage
- **Fixtures**: Use pytest fixtures for test data
- **Mocking**: Mock external dependencies appropriately
- **Assertions**: Use descriptive assertions with clear messages

### Test Organization
```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for API and database
├── e2e/           # End-to-end tests for complete workflows
├── performance/   # Performance and benchmark tests
└── fixtures/      # Shared test fixtures and data
```

### Test Patterns
- **Arrange-Act-Assert**: Clear test structure
- **Given-When-Then**: BDD-style test descriptions
- **Parametrized Tests**: Test multiple scenarios efficiently
- **Property-Based Testing**: Test with generated data

### Test Data
- **Fixtures**: Reusable test data and objects
- **Factories**: Generate test data dynamically
- **Mocking**: Mock external services and dependencies
- **Database**: Use test database with proper cleanup

### Test Execution
- **Fast**: Unit tests should run quickly
- **Isolated**: Tests should not depend on each other
- **Repeatable**: Tests should produce consistent results
- **Automated**: All tests should run as part of an automated pipeline

Always generate comprehensive tests that follow these standards.
"""
    
    def _get_documentation_standards_rule(self) -> str:
        """Get documentation standards rule for Cursor AI."""
        return """# Documentation Standards for AI Coding Agents

## 📚 Comprehensive Documentation Requirements

### Code Documentation
- **Docstrings**: Google-style docstrings for all public functions
- **Type Hints**: Comprehensive type annotations
- **Comments**: Inline comments for complex business logic
- **Examples**: Code examples in docstrings and documentation

### API Documentation
- **OpenAPI**: Auto-generated API documentation
- **Endpoints**: Clear endpoint descriptions and examples
- **Schemas**: Detailed request/response schemas
- **Authentication**: Clear authentication requirements

### Project Documentation
- **README**: Comprehensive project overview and setup
- **Architecture**: System design and architecture decisions
- **Deployment**: Production deployment instructions
- **Contributing**: Development workflow and standards

### Documentation Structure
```
docs/
├── README.md              # Project overview and quick start
├── source/               # Sphinx documentation source
│   ├── installation.rst  # Installation guide
│   ├── quickstart.rst    # Quick start guide
│   ├── api/             # API documentation
│   ├── development/     # Development documentation
│   └── architecture/    # Architecture documentation
└── build/               # Generated documentation
```

### Documentation Quality
- **Accuracy**: Documentation must be accurate and up-to-date
- **Completeness**: Cover all public APIs and features
- **Examples**: Include practical examples and use cases
- **Clarity**: Use clear, concise language

### Auto-Generation
- **API Docs**: Auto-generate from code annotations
- **Code Coverage**: Auto-generate coverage reports
- **Dependencies**: Auto-generate dependency reports
- **Changelog**: Auto-generate from git commits

Always generate and maintain comprehensive documentation.
"""


def main():
    """Main function to setup AI context."""
    project_root = Path.cwd()
    
    setup = AIContextSetup(project_root)
    success = setup.setup_ai_context()
    
    if success:
        print("\n🎉 AI Coding Agent setup complete!")
        print("\nNext steps:")
        print("1. Install pre-commit hooks: pre-commit install")
        print("2. Install development dependencies: pip install -e '.[dev]'")
        print("3. Run tests: pytest")
        print("4. Generate documentation: sphinx-build docs/source docs/build")
        print("\nYour project is now ready for AI-assisted development!")
        return 0
    else:
        print("\n❌ AI setup failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
