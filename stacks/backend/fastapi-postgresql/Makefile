# Makefile for FastAPI PostgreSQL Backend Template
# Provides comprehensive automation for backend development workflows

.PHONY: help install install-dev test test-unit test-integration test-e2e test-performance lint format type-check security clean build run dev docker-build docker-run docker-dev docs migrate db-shell api-docs health check ci setup

# Default target
help: ## Show this help message
	@echo "FastAPI PostgreSQL Backend - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e ".[dev]"
	pre-commit install

# Database
db-init: ## Initialize database
	python -c "import asyncio; from src.app.core.database import init_database; asyncio.run(init_database())"

db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migrations
	alembic downgrade -1

db-revision: ## Create new database migration
	alembic revision --autogenerate -m "$(message)"

db-reset: ## Reset database (WARNING: Deletes all data)
	alembic downgrade base
	alembic upgrade head

db-shell: ## Connect to database shell
	@echo "Connecting to database..."
	@if command -v psql >/dev/null 2>&1; then \
		psql $(shell python -c "from src.app.core.config import get_settings; print(get_settings().SQLALCHEMY_DATABASE_URI.replace('postgresql+asyncpg://', 'postgresql://'))"); \
	else \
		echo "psql not found. Install PostgreSQL client tools."; \
	fi

db-health: ## Check database connectivity
	python -c "import asyncio; from src.app.core.database import check_database_health; result = asyncio.run(check_database_health()); print('Database health:', result)"

# Testing
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v --tb=short

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v --tb=short

test-performance: ## Run performance tests only
	pytest tests/performance/ --benchmark-only

test-coverage: ## Run tests with coverage report
	pytest --cov=src/app --cov-report=html --cov-report=term-missing

test-auth: ## Run authentication-related tests
	pytest tests/ -k "auth" -v

test-database: ## Run database-related tests
	pytest tests/ -k "database" -v

test-api: ## Run API-related tests
	pytest tests/ -k "api" -v

# Code Quality
lint: ## Run linting
	flake8 src/app tests
	pylint src/app

format: ## Format code
	black src/app tests
	isort src/app tests

format-check: ## Check code formatting
	black --check src/app tests
	isort --check-only src/app tests

type-check: ## Run type checking
	mypy src/app

# Security
security: ## Run security checks
	bandit -r src/app/
	safety check
	pip-audit

security-full: ## Run comprehensive security checks
	bandit -r src/app/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

# Development
dev: ## Start development server
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

dev-debug: ## Start development server with debug logging
	uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

run: ## Start production server
	uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 4

shell: ## Start Python shell with app context
	python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from app.core.database import get_async_session_maker
async def setup():
    # Setup database session for interactive use
    pass
asyncio.run(setup())
import code
code.interact(local=globals())
"

# Docker
docker-build: ## Build Docker image
	docker build -t fastapi-postgresql-backend:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 fastapi-postgresql-backend:latest

docker-dev: ## Run development environment with Docker Compose
	docker-compose -f docker/docker-compose.dev.yml up --build

docker-test: ## Run tests in Docker
	docker build -t fastapi-backend-test .
	docker run --rm --env-file .env fastapi-backend-test pytest tests/ -v

docker-clean: ## Clean up Docker resources
	docker-compose down -v
	docker system prune -f

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/

docs-serve: ## Serve documentation locally
	cd docs/_build && python -m http.server 8000

docs-clean: ## Clean documentation build
	rm -rf docs/_build/

# API Documentation
api-docs: ## Generate OpenAPI specification
	python -c "
import sys
sys.path.insert(0, 'src')
from app.main import app
import json
with open('openapi-spec.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
print('OpenAPI spec generated: openapi-spec.json')
"

api-docs-serve: ## Serve API documentation
	python -c "
import sys
sys.path.insert(0, 'src')
from app.main import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
" &

# Monitoring
logs: ## View application logs
	docker-compose logs -f fastapi-app

health: ## Check application health
	curl -f http://localhost:8000/health/live && echo "✅ API is healthy" || echo "❌ API is unhealthy"

health-db: ## Check database health
	python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from app.core.database import check_database_health
result = asyncio.run(check_database_health())
print('Database status:', '✅ healthy' if result['status'] == 'healthy' else '❌ unhealthy')
"

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
	rm -rf *.log logs/
	rm -rf test-results/ coverage-reports/

clean-all: clean ## Clean everything including virtual environment
	rm -rf venv/ .venv/ env/
	rm -rf bandit-report.json safety-report.json pip-audit-report.json

# Build and Distribution
build: ## Build package
	python -m build

dist: ## Create distribution packages
	python -m build --wheel --sdist

# Pre-commit
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Environment
env: ## Create .env file from template
	cp env.example .env

env-check: ## Check environment variables
	python -c "
import sys
sys.path.insert(0, 'src')
from app.core.config import get_settings
settings = get_settings()
print('✅ Environment variables loaded successfully')
print(f'DEBUG: {settings.DEBUG}')
print(f'DATABASE: {settings.SQLALCHEMY_DATABASE_URI.split('@')[0]}@***')
"

# Performance
benchmark: ## Run performance benchmarks
	pytest tests/performance/ --benchmark-only --benchmark-save=benchmark

benchmark-compare: ## Compare with previous benchmark
	pytest tests/performance/ --benchmark-compare

# CI/CD helpers
ci-test: ## Run tests for CI/CD
	pytest --cov=src/app --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml

ci-lint: ## Run linting for CI/CD
	black --check src/app tests
	isort --check-only src/app tests
	flake8 src/app tests
	mypy src/app

ci-security: ## Run security checks for CI/CD
	bandit -r src/app/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

ci-migrate: ## Run database migrations for CI/CD
	alembic upgrade head

# Development workflow
setup: install-dev env db-init db-upgrade ## Complete development setup
	@echo "🚀 FastAPI PostgreSQL Backend setup complete!"
	@echo "Run 'make dev' to start the development server"
	@echo "Run 'make test' to run the test suite"

check: format-check lint type-check security ## Run all checks
	@echo "✅ All checks passed!"

ci: ci-lint ci-test ci-security ci-migrate ## Run CI/CD pipeline locally
	@echo "✅ CI/CD pipeline completed successfully!"

# Quick development commands
quick-test: ## Quick test run
	pytest tests/unit/ -x --tb=short

quick-format: ## Quick format check
	black --check src/app tests --quiet

quick-lint: ## Quick lint check
	flake8 src/app tests --max-line-length=88 --extend-ignore=E203,W503 --quiet

# API testing
api-test: ## Test API endpoints
	@echo "Testing API endpoints..."
	@curl -s http://localhost:8000/health/live >/dev/null && echo "✅ Health check passed" || echo "❌ Health check failed"
	@curl -s http://localhost:8000/docs >/dev/null && echo "✅ Docs endpoint accessible" || echo "❌ Docs endpoint not accessible"

# Database seeding
seed: ## Seed database with sample data
	python scripts/seed_database.py

# User management
create-superuser: ## Create superuser account
	@echo "Creating superuser account..."
	@python -c "
import asyncio
import sys
import getpass
sys.path.insert(0, 'src')
from app.services.user_service import UserService
from app.core.database import get_db
from app.schemas.user import UserCreate

async def create_admin():
    email = input('Email: ')
    username = input('Username: ')
    password = getpass.getpass('Password: ')

    user_data = UserCreate(
        email=email,
        username=username,
        password=password
    )

    async for session in get_db():
        service = UserService(session)
        user = await service.create_user(user_data)
        # Make superuser
        user.is_superuser = True
        user.email_verified = True
        await session.commit()
        print(f'✅ Superuser {user.username} created successfully!')

asyncio.run(create_admin())
"

# Version management
version: ## Show current version
	python -c "
import sys
sys.path.insert(0, 'src')
from app import __version__
print(__version__)
"

# Release
release: clean test check build ## Create a release
	@echo "🎉 Release created successfully!"
	@echo "Run 'pip install dist/*.whl' to install locally"

# Custom backend commands
routes: ## List all API routes
	python -c "
import sys
sys.path.insert(0, 'src')
from app.main import app
for route in app.routes:
    if hasattr(route, 'methods') and hasattr(route, 'path'):
        methods = ', '.join(route.methods)
        print(f'{methods:8} {route.path}')
"

deps: ## Show dependency tree
	pipdeptree

deps-outdated: ## Show outdated dependencies
	pip list --outdated

# Monitoring and debugging
profile: ## Profile application performance
	python -c "
import sys
sys.path.insert(0, 'src')
from app.main import app
import uvicorn
# Add profiling middleware here
print('Profiling enabled - start server with: make dev')
"

debug-db: ## Debug database queries
	python -c "
import sys
sys.path.insert(0, 'src')
from app.core.database import get_engine
engine = get_engine()
print('Database engine:', engine)
print('Pool size:', engine.pool.size)
print('Checked out connections:', engine.pool.checkedout())
"
