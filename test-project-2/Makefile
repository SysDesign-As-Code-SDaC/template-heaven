# Makefile for test-project-2
# Provides common development tasks and commands

.PHONY: help install install-dev test test-unit test-integration test-e2e test-performance lint format type-check security clean build run dev docker-build docker-run docker-dev docker-test docs serve-docs

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v

test-performance: ## Run performance tests only
	pytest tests/performance/ --benchmark-only

test-coverage: ## Run tests with coverage report
	pytest --cov=app --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	pytest-watch

# Code Quality
lint: ## Run linting
	flake8 app tests
	pylint app

format: ## Format code
	black app tests
	isort app tests

format-check: ## Check code formatting
	black --check app tests
	isort --check-only app tests

type-check: ## Run type checking
	mypy app

# Security
security: ## Run security checks
	bandit -r app/
	safety check
	pip-audit

security-full: ## Run comprehensive security checks
	bandit -r app/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

# Database
db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migrations
	alembic downgrade -1

db-revision: ## Create new database migration
	alembic revision --autogenerate -m "$(message)"

db-reset: ## Reset database (WARNING: This will delete all data)
	alembic downgrade base
	alembic upgrade head

# Development
dev: ## Start development server
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run: ## Start production server
	uvicorn app.main:app --host 0.0.0.0 --port 8000

shell: ## Start Python shell with app context
	python -c "from app.main import app; import asyncio; asyncio.run(app.router.startup())"

# Docker
docker-build: ## Build Docker image
	docker build -t test-project-2:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 test-project-2:latest

docker-dev: ## Run development environment with Docker Compose
	docker-compose --profile dev up --build

docker-test: ## Run tests in Docker
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

docker-clean: ## Clean up Docker resources
	docker-compose down -v
	docker system prune -f

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/source docs/build

docs-serve: ## Serve documentation locally
	cd docs/build && python -m http.server 8000

docs-clean: ## Clean documentation build
	rm -rf docs/build

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .bandit/
	rm -rf .safety/

clean-all: clean ## Clean everything including virtual environment
	rm -rf venv/
	rm -rf .venv/

# Build and Distribution
build: ## Build package
	python -m build

dist: ## Create distribution packages
	python -m build --wheel --sdist

upload-test: ## Upload to test PyPI
	twine upload --repository testpypi dist/*

upload: ## Upload to PyPI
	twine upload dist/*

# Pre-commit
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Environment
env: ## Create .env file from template
	cp .env.example .env

env-check: ## Check environment variables
	python -c "from app.core.config import get_settings; print('Environment check passed')"

# Monitoring
logs: ## View application logs
	docker-compose logs -f test-project-2

logs-db: ## View database logs
	docker-compose logs -f postgres

logs-redis: ## View Redis logs
	docker-compose logs -f redis

# Health checks
health: ## Check application health
	curl -f http://localhost:8000/health/live

health-ready: ## Check application readiness
	curl -f http://localhost:8000/health/ready

# Database management
db-shell: ## Connect to database shell
	docker-compose exec postgres psql -U test-project-2 -d test-project-2_db

db-backup: ## Backup database
	docker-compose exec postgres pg_dump -U test-project-2 test-project-2_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup
	@echo "Usage: make db-restore BACKUP_FILE=backup_file.sql"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE"; exit 1; fi
	docker-compose exec -T postgres psql -U test-project-2 -d test-project-2_db < $(BACKUP_FILE)

# Performance
benchmark: ## Run performance benchmarks
	pytest tests/performance/ --benchmark-only --benchmark-save=benchmark

benchmark-compare: ## Compare with previous benchmark
	pytest tests/performance/ --benchmark-compare

# CI/CD helpers
ci-test: ## Run tests for CI/CD
	pytest --cov=app --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml

ci-lint: ## Run linting for CI/CD
	black --check app tests
	isort --check-only app tests
	flake8 app tests
	mypy app

ci-security: ## Run security checks for CI/CD
	bandit -r app/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

# Development workflow
setup: install-dev env ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "Run 'make dev' to start the development server"

check: format-check lint type-check security ## Run all checks
	@echo "All checks passed!"

ci: ci-lint ci-test ci-security ## Run CI/CD pipeline locally
	@echo "CI/CD pipeline completed successfully!"

# Quick development commands
quick-test: ## Quick test run
	pytest tests/unit/ -x

quick-format: ## Quick format check
	black --check app tests
	isort --check-only app tests

quick-lint: ## Quick lint check
	flake8 app tests --max-line-length=88 --extend-ignore=E203,W503

# Database seeding
seed: ## Seed database with sample data
	python scripts/seed_database.py

# API documentation
api-docs: ## Generate API documentation
	python -c "from app.main import app; import json; print(json.dumps(app.openapi(), indent=2))" > api-spec.json

# Version management
version: ## Show current version
	python -c "from app import __version__; print(__version__)"

version-bump: ## Bump version (usage: make version-bump TYPE=patch|minor|major)
	@if [ -z "$(TYPE)" ]; then echo "Please specify TYPE (patch|minor|major)"; exit 1; fi
	bump2version $(TYPE)

# Release
release: clean test check build ## Create a release
	@echo "Release created successfully!"
	@echo "Run 'make upload' to upload to PyPI"
