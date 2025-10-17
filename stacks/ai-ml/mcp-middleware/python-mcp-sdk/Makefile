# Python MCP SDK Template - Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install dev-install test lint format clean build run-server run-client docker-build docker-up docker-down docker-logs

# Default target
help: ## Show this help message
	@echo "Python MCP SDK Template - Available Commands:"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# Development Setup
# ============================================================================

install: ## Install production dependencies
	pip install -r requirements.txt

dev-install: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

setup: ## Complete development setup
	@echo "Setting up development environment..."
	cp env.example .env
	@echo "✓ Environment file created (edit .env as needed)"
	$(MAKE) dev-install
	@echo "✓ Dependencies installed"
	@echo "✓ Development setup complete!"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run linting checks
	@echo "Running linting checks..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports
	bandit -r src/ -f json -o bandit-report.json || true
	@echo "✓ Linting complete"

format: ## Format code with black and isort
	@echo "Formatting code..."
	black src/ tests/ --line-length=88
	isort src/ tests/ --profile=black
	@echo "✓ Code formatted"

format-check: ## Check code formatting without making changes
	@echo "Checking code formatting..."
	black src/ tests/ --check --line-length=88
	isort src/ tests/ --check-only --profile=black
	@echo "✓ Code formatting check complete"

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo "Running tests..."
	pytest tests/ -v --cov=src/ --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src/ --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# ============================================================================
# Application Commands
# ============================================================================

run-server: ## Start the MCP server
	@echo "Starting MCP server..."
	python -m src.mcp_sdk.main server

run-client: ## Start the MCP client
	@echo "Starting MCP client..."
	python -m src.mcp_sdk.main client

run-test: ## Run connectivity tests
	@echo "Running connectivity tests..."
	python -m src.mcp_sdk.main test

info: ## Show application information
	@echo "Application information:"
	python -m src.mcp_sdk.main info

# ============================================================================
# Docker Commands
# ============================================================================

docker-build: ## Build Docker images
	@echo "Building Docker images..."
	docker build -t mcp-sdk:latest .
	docker build -f Dockerfile.client -t mcp-sdk-client:latest .
	docker build -f Dockerfile.jupyter -t mcp-sdk-jupyter:latest .

docker-up: ## Start services with Docker Compose
	@echo "Starting services with Docker Compose..."
	docker-compose up -d

docker-up-dev: ## Start development services with Docker Compose
	@echo "Starting development services..."
	docker-compose --profile dev up -d

docker-down: ## Stop Docker Compose services
	@echo "Stopping Docker Compose services..."
	docker-compose down

docker-logs: ## Show Docker Compose logs
	@echo "Showing Docker Compose logs..."
	docker-compose logs -f

docker-clean: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f

# ============================================================================
# Database Commands
# ============================================================================

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	alembic upgrade head

db-revision: ## Create new database revision
	@echo "Creating new database revision..."
	alembic revision --autogenerate -m "$(MESSAGE)"

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		alembic downgrade base; \
		alembic upgrade head; \
		echo "✓ Database reset complete"; \
	else \
		echo "Database reset cancelled"; \
	fi

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation
	@echo "Generating documentation..."
	mkdocs build
	@echo "✓ Documentation generated in site/"

docs-serve: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	mkdocs serve

# ============================================================================
# Security
# ============================================================================

security-scan: ## Run security scans
	@echo "Running security scans..."
	bandit -r src/ -f json -o bandit-report.json
	safety check
	@echo "✓ Security scan complete"

# ============================================================================
# Performance
# ============================================================================

benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	python -m pytest tests/benchmarks/ -v --benchmark-only

profile: ## Profile application performance
	@echo "Profiling application performance..."
	python -m cProfile -o profile.stats src/mcp_sdk/main.py server &
	sleep 10
	pkill -f "python -m cProfile"
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Clean up generated files
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf bandit-report.json
	rm -rf profile.stats
	@echo "✓ Cleanup complete"

# ============================================================================
# Release
# ============================================================================

release: ## Create a new release
	@echo "Creating new release..."
	@read -p "Enter version number: " version; \
	git tag -a v$$version -m "Release v$$version"; \
	git push origin v$$version; \
	echo "✓ Release v$$version created"

# ============================================================================
# Development Workflow
# ============================================================================

dev: ## Start development environment
	@echo "Starting development environment..."
	$(MAKE) docker-up-dev
	@echo "✓ Development environment started"
	@echo "  - MCP Server: http://localhost:8000"
	@echo "  - Jupyter: http://localhost:8888"
	@echo "  - PgAdmin: http://localhost:5050"
	@echo "  - Redis Commander: http://localhost:8081"

dev-stop: ## Stop development environment
	@echo "Stopping development environment..."
	$(MAKE) docker-down
	@echo "✓ Development environment stopped"

# ============================================================================
# Production Deployment
# ============================================================================

deploy: ## Deploy to production
	@echo "Deploying to production..."
	$(MAKE) docker-build
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✓ Production deployment complete"

# ============================================================================
# Monitoring
# ============================================================================

monitor: ## Start monitoring stack
	@echo "Starting monitoring stack..."
	docker-compose --profile monitoring up -d
	@echo "✓ Monitoring stack started"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3001 (admin/admin123)"

# ============================================================================
# Quick Commands
# ============================================================================

quick-test: ## Quick test of the system
	@echo "Running quick system test..."
	$(MAKE) run-server &
	sleep 5
	$(MAKE) run-test
	pkill -f "python -m src.mcp_sdk.main server"
	@echo "✓ Quick test complete"

all-checks: ## Run all quality checks
	@echo "Running all quality checks..."
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) security-scan
	@echo "✓ All quality checks complete"
