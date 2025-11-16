# FastAPI PostgreSQL Backend Dockerfile
# Multi-stage build for optimal image size and performance

# =============================================================================
# Base stage with Python and system dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libpq-dev \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN groupadd -r fastapi && useradd -r -g fastapi fastapi

# Set working directory
WORKDIR /app

# =============================================================================
# Dependencies stage - install Python packages
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Development stage - full development environment
# =============================================================================
FROM dependencies as development

# Install additional development tools
RUN pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel

# Setup Jupyter kernel
RUN python -m ipykernel install --user --name fastapi-backend --display-name "FastAPI Backend"

# Create directories
RUN mkdir -p /app/data /app/logs /app/uploads && \
    chown -R fastapi:fastapi /app

# Copy application code
COPY . .

# Set permissions
RUN chown -R fastapi:fastapi /app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Production stage - minimal runtime environment
# =============================================================================
FROM dependencies as production

# Install only runtime dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard]

# Create directories
RUN mkdir -p /app/data /app/logs /app/uploads && \
    chown -R fastapi:fastapi /app

# Copy application code (excluding development files)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./
COPY alembic/ ./alembic/
COPY env.example ./

# Set permissions
RUN chown -R fastapi:fastapi /app

# Switch to non-root user
USER fastapi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command with gunicorn for production
CMD ["gunicorn", \
     "src.app.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# =============================================================================
# Testing stage - for automation pipelines (CI/CD examples disabled)
# =============================================================================
FROM dependencies as testing

# Install testing dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Create directories
RUN mkdir -p /app/test-results /app/coverage-reports && \
    chown -R fastapi:fastapi /app

# Copy application code
COPY . .

# Set permissions
RUN chown -R fastapi:fastapi /app

# Switch to non-root user
USER fastapi

# Default command for testing
CMD ["pytest", "tests/", "-v", "--cov=src/app", "--cov-report=html", "--cov-report=term"]

# =============================================================================
# Migration stage - for database migrations
# =============================================================================
FROM dependencies as migration

# Install alembic
RUN pip install --no-cache-dir alembic

# Create directories
RUN mkdir -p /app/alembic/versions && \
    chown -R fastapi:fastapi /app

# Copy migration files
COPY alembic/ ./alembic/
COPY src/ ./src/
COPY pyproject.toml ./

# Set permissions
RUN chown -R fastapi:fastapi /app

# Switch to non-root user
USER fastapi

# Default command for migrations
CMD ["alembic", "upgrade", "head"]

# =============================================================================
# Default target - development environment
# =============================================================================
FROM development
