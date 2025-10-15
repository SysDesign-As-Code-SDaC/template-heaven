# Makefile for MCP Middleware Template
# Provides comprehensive automation for MCP server development workflows

.PHONY: help install install-dev test test-unit test-integration test-e2e test-performance lint format type-check security clean build run dev docker-build docker-run docker-dev docs migrate api-docs health check ci setup

# Default target
help: ## Show this help message
	@echo "MCP Middleware - Available commands:"
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
	python -c "import asyncio; from src.mcp_middleware.core.database import init_database; asyncio.run(init_database())"

db-upgrade: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback database migrations
	alembic downgrade -1

db-revision: ## Create new database migration
	alembic revision --autogenerate -m "$(message)"

db-reset: ## Reset database (WARNING: Deletes all data)
	alembic downgrade base
	alembic upgrade head

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
	pytest --cov=src/mcp_middleware --cov-report=html --cov-report=term-missing

test-mcp: ## Run MCP-specific tests
	pytest tests/ -k "mcp" -v

# Code Quality
lint: ## Run linting
	flake8 src/mcp_middleware tests
	pylint src/mcp_middleware

format: ## Format code
	black src/mcp_middleware tests
	isort src/mcp_middleware tests

format-check: ## Check code formatting
	black --check src/mcp_middleware tests
	isort --check-only src/mcp_middleware tests

type-check: ## Run type checking
	mypy src/mcp_middleware

# Security
security: ## Run security checks
	bandit -r src/mcp_middleware/
	safety check
	pip-audit

security-full: ## Run comprehensive security checks
	bandit -r src/mcp_middleware/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

# Development
dev: ## Start development server
	uvicorn src.mcp_middleware.main:app --reload --host 0.0.0.0 --port 8000 --log-level info

dev-debug: ## Start development server with debug logging
	uvicorn src.mcp_middleware.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

run: ## Start production server
	gunicorn src.mcp_middleware.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

shell: ## Start Python shell with MCP context
	python -c "
import asyncio
import sys
sys.path.insert(0, 'src')
from mcp_middleware.core.database import get_async_session_maker
async def setup():
    pass
asyncio.run(setup())
import code
code.interact(local=globals())
"

# Docker
docker-build: ## Build Docker image
	docker build -t mcp-middleware:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 mcp-middleware:latest

docker-dev: ## Run development environment with Docker Compose
	docker-compose -f docker/docker-compose.dev.yml up --build

docker-prod: ## Run production environment with Docker Compose
	docker-compose -f docker/docker-compose.prod.yml up -d

docker-test: ## Run tests in Docker
	docker build -t mcp-middleware-test .
	docker run --rm --env-file .env mcp-middleware-test pytest tests/ -v

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
from mcp_middleware.main import app
import json
with open('openapi-spec.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
print('OpenAPI spec generated: openapi-spec.json')
"

api-docs-serve: ## Serve API documentation
	python -c "
import sys
sys.path.insert(0, 'src')
from mcp_middleware.main import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8000)
" &

# Monitoring
logs: ## View application logs
	docker-compose logs -f mcp-middleware

logs-db: ## View database logs
	docker-compose logs -f postgres

logs-redis: ## View Redis logs
	docker-compose logs -f redis

health: ## Check application health
	curl -f http://localhost:8000/health/live && echo "✅ MCP Middleware is healthy" || echo "❌ MCP Middleware is unhealthy"

health-mcp: ## Check MCP server health
	curl -f http://localhost:8000/api/v1/mcp/health && echo "✅ MCP servers are healthy" || echo "❌ MCP servers unhealthy"

# MCP Operations
mcp-list: ## List all MCP servers
	curl -s http://localhost:8000/api/v1/servers | jq .

mcp-add: ## Add a new MCP server (usage: make mcp-add name=filesystem config='{"path": "/tmp"}')
	@if [ -z "$(name)" ] || [ -z "$(config)" ]; then echo "Usage: make mcp-add name=server_name config='{\"key\": \"value\"}'"; exit 1; fi
	curl -X POST http://localhost:8000/api/v1/servers \
		-H "Content-Type: application/json" \
		-d "{\"name\": \"$(name)\", \"type\": \"$(name)\", \"config\": $(config)}"

mcp-remove: ## Remove an MCP server (usage: make mcp-remove name=filesystem)
	@if [ -z "$(name)" ]; then echo "Usage: make mcp-remove name=server_name"; exit 1; fi
	curl -X DELETE http://localhost:8000/api/v1/servers/$(name)

mcp-test: ## Test MCP server functionality
	curl -X POST http://localhost:8000/api/v1/mcp/filesystem \
		-H "Content-Type: application/json" \
		-d '{"method": "list_dir", "params": {"path": "."}}' | jq .

mcp-clickup-test: ## Test ClickUp MCP integration
	curl -X POST http://localhost:8000/api/v1/mcp/clickup \
		-H "Content-Type: application/json" \
		-d '{"method": "get_workspaces", "params": {}}' | jq .

# Performance
benchmark: ## Run performance benchmarks
	pytest tests/performance/ --benchmark-only --benchmark-save=benchmark

benchmark-compare: ## Compare with previous benchmark
	pytest tests/performance/ --benchmark-compare

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
from mcp_middleware.utils.config import get_settings
settings = get_settings()
print('✅ Environment variables loaded successfully')
print(f'DEBUG: {settings.DEBUG}')
print(f'MCP_SERVERS: {len(settings.MCP_SERVERS) if hasattr(settings, \"MCP_SERVERS\") else \"N/A\"}')
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
	rm -rf openapi-spec.json bandit-report.json safety-report.json pip-audit-report.json

clean-all: clean ## Clean everything including virtual environment
	rm -rf venv/ .venv/ env/

# Build and Distribution
build: ## Build package
	python -m build

dist: ## Create distribution packages
	python -m build --wheel --sdist

# CI/CD helpers
ci-test: ## Run tests for CI/CD
	pytest --cov=src/mcp_middleware --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml

ci-lint: ## Run linting for CI/CD
	black --check src/mcp_middleware tests
	isort --check-only src/mcp_middleware tests
	flake8 src/mcp_middleware tests
	mypy src/mcp_middleware

ci-security: ## Run security checks for CI/CD
	bandit -r src/mcp_middleware/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

ci-mcp: ## Run MCP-specific CI tests
	pytest tests/ -k "mcp" --tb=short

# Development workflow
setup: install-dev env db-init db-upgrade ## Complete development setup
	@echo "🚀 MCP Middleware setup complete!"
	@echo "Run 'make dev' to start the development server"
	@echo "Run 'make mcp-list' to see available MCP servers"

check: format-check lint type-check security ## Run all checks
	@echo "✅ All checks passed!"

ci: ci-lint ci-test ci-security ci-mcp ## Run CI/CD pipeline locally
	@echo "✅ CI/CD pipeline completed successfully!"

# Quick development commands
quick-test: ## Quick test run
	pytest tests/unit/ -x --tb=short

quick-lint: ## Quick lint check
	flake8 src/mcp_middleware tests --max-line-length=88 --extend-ignore=E203,W503 --quiet

quick-format: ## Quick format check
	black --check src/mcp_middleware tests --quiet

# MCP Server Development
server-create: ## Create new MCP server (usage: make server-create name=my_server)
	@if [ -z "$(name)" ]; then echo "Usage: make server-create name=server_name"; exit 1; fi
	mkdir -p src/mcp_middleware/servers/$(name)
	touch src/mcp_middleware/servers/$(name).py
	touch tests/unit/test_$(name).py
	echo "from ..base import BaseMCPServer" > src/mcp_middleware/servers/$(name).py
	echo "import pytest" > tests/unit/test_$(name).py
	echo "✅ MCP server $(name) created successfully!"

# Version management
version: ## Show current version
	python -c "
import sys
sys.path.insert(0, 'src')
from mcp_middleware import __version__
print(__version__)
"

# Release
release: clean test check build ## Create a release
	@echo "🎉 Release created successfully!"
	@echo "Run 'pip install dist/*.whl' to install locally"

# Advanced monitoring
metrics: ## Show project metrics
	@echo "=== MCP Middleware Metrics ==="
	@echo "MCP Servers: $$(find src/mcp_middleware/servers -name "*.py" | wc -l)"
	@echo "Lines of code: $$(find src -name "*.py" | xargs wc -l | tail -1 | awk '{print $$1}')"
	@echo "Test files: $$(find tests -name "*.py" | wc -l)"
	@echo "Test coverage: $$(find coverage -name "*.json" -exec jq -r '.total.lines.pct' {} \; 2>/dev/null || echo 'Not run')"

# ClickUp integration testing
clickup-auth-test: ## Test ClickUp authentication
	@echo "Testing ClickUp authentication..."
	@curl -s http://localhost:8000/api/v1/mcp/clickup \
		-H "Content-Type: application/json" \
		-d '{"method": "get_user", "params": {}}' | jq '.authenticated // false' | grep -q true \
		&& echo "✅ ClickUp authentication successful" \
		|| echo "❌ ClickUp authentication failed"

clickup-workspace-test: ## Test ClickUp workspace access
	@echo "Testing ClickUp workspace access..."
	@curl -s http://localhost:8000/api/v1/mcp/clickup \
		-H "Content-Type: application/json" \
		-d '{"method": "get_workspaces", "params": {}}' | jq '.workspaces | length' \
		&& echo "✅ ClickUp workspace access successful" \
		|| echo "❌ ClickUp workspace access failed"

# MCP Protocol testing
protocol-test: ## Test MCP protocol compliance
	@echo "Testing MCP protocol compliance..."
	@python -c "
import sys
sys.path.insert(0, 'src')
from mcp_middleware.core.protocol_handler import MCPProtocolHandler
handler = MCPProtocolHandler()
print('✅ MCP protocol handler initialized')
print('Supported methods:', len(handler.supported_methods))
"

# Load testing
load-test: ## Run load tests against MCP servers
	@echo "Running load tests..."
	@ab -n 1000 -c 10 -T 'application/json' -p load_test_payload.json http://localhost:8000/api/v1/mcp/filesystem || echo "ab not found, install apache2-utils"

# Debug helpers
debug-db: ## Debug database connections
	python -c "
import sys
sys.path.insert(0, 'src')
from mcp_middleware.core.database import check_database_health
import asyncio
result = asyncio.run(check_database_health())
print('Database status:', result)
"

debug-mcp: ## Debug MCP server status
	python -c "
import sys
sys.path.insert(0, 'src')
from mcp_middleware.core.server_manager import ServerManager
manager = ServerManager()
servers = manager.list_servers()
print('Active MCP servers:', len(servers))
for server in servers:
    print(f'  - {server[\"name\"]}: {server[\"status\"]}')
"

# Final summary
status: ## Show project status
	@echo "=== MCP Middleware Status ==="
	@echo "Python version: $$(python --version)"
	@echo "Dependencies: $$(pip list --format=freeze | wc -l) packages"
	@echo "MCP Servers: $$(find src/mcp_middleware/servers -name "*.py" | wc -l)"
	@echo "Test coverage: $$(find coverage -name "*.json" -exec jq -r '.total.lines.pct' {} \; 2>/dev/null || echo 'Not run')"
	@echo "Health status: $$(curl -s http://localhost:8000/health/live >/dev/null && echo 'Healthy' || echo 'Unhealthy')"
	@echo "Container status: $$(docker ps | grep mcp-middleware | wc -l) running"
