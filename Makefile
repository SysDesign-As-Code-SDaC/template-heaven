# Template Heaven - Development Makefile
# This Makefile provides common development tasks for the Template Heaven package

.PHONY: help install install-dev test test-cov lint format clean build publish docs

# Default target
help:
	@echo "Template Heaven - Available Commands:"
	@echo ""
	@echo "  install      Install package in development mode"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  build        Build package"
	@echo "  publish      Publish package to PyPI"
	@echo "  docs         Generate documentation"
	@echo "  check        Run all checks (lint, test, format)"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=templateheaven --cov-report=html --cov-report=term-missing

# Linting and formatting
lint:
	flake8 templateheaven tests
	mypy templateheaven

format:
	black templateheaven tests
	isort templateheaven tests

format-check:
	black --check templateheaven tests
	isort --check-only templateheaven tests

# Code quality checks
check: format-check lint test

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build and publish
build: clean
	python -m build

publish: build
	twine upload dist/*

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Development setup
setup: install-dev
	@echo "Development environment setup complete!"

# Run the CLI
run:
	python -m templateheaven.cli.main

# Run with specific command
init:
	python -m templateheaven.cli.main init

list:
	python -m templateheaven.cli.main list

# Automation helpers: Disabled
ci-install:
	@echo "Automation helpers disabled by maintainer"

ci-test:
	@echo "Automation tests disabled by maintainer"

ci-lint:
	@echo "Automation lint disabled by maintainer"

# Template management (for development)
update-templates:
	@echo "Template update not yet implemented"

validate-templates:
	@echo "Template validation not yet implemented"

# Cache management
clear-cache:
	python -c "from templateheaven.config.settings import Config; Config().get_cache_dir().rmdir() if Config().get_cache_dir().exists() else None; print('Cache cleared')"

# Version management
version:
	@python -c "from templateheaven import __version__; print(__version__)"

# Security checks
security:
	bandit -r templateheaven/
	safety check

# Performance testing
benchmark:
	@echo "Performance benchmarking not yet implemented"

# Docker helpers (for future use)
docker-build:
	@echo "Docker build not yet implemented"

docker-test:
	@echo "Docker testing not yet implemented"

# Release helpers
release-check:
	@echo "Checking release readiness..."
	@python -c "import templateheaven; print(f'Version: {templateheaven.__version__}')"
	@echo "Running tests..."
	@make test
	@echo "Running linting..."
	@make lint
	@echo "Checking format..."
	@make format-check
	@echo "Release check complete!"

# Help for specific targets
install-help:
	@echo "Installation Commands:"
	@echo "  make install      - Install package in development mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make setup        - Complete development setup"

test-help:
	@echo "Testing Commands:"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make ci-test      - (disabled)"

lint-help:
	@echo "Linting Commands:"
	@echo "  make lint         - Run flake8 and mypy"
	@echo "  make format       - Format code with black and isort"
	@echo "  make format-check - Check code formatting"
	@echo "  make check        - Run all quality checks"

# Environment info
env-info:
	@echo "Environment Information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Package version: $(shell python -c 'from templateheaven import __version__; print(__version__)')"

# Quick development cycle
dev: format lint test
	@echo "Development cycle complete!"

# Yeoman generator helpers
generator-install:
	@echo "Installing Yeoman generator dependencies..."
	cd generators/templateheaven-generator && npm install

generator-run:
	@echo "Run the Yeoman generator from the target directory (e.g., mkdir newapp && cd newapp && make generator-run)"
	@cd generators/templateheaven-generator && ./node_modules/.bin/yo templateheaven

export-stacks-json:
	@python scripts/export_stacks_json.py

# Full development setup
dev-setup: clean install-dev format lint test
	@echo "Full development setup complete!"
