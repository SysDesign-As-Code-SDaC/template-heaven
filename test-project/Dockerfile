# Multi-stage Docker build for test-project
# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for building
RUN groupadd -r builder && useradd -r -g builder builder

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install --no-cache-dir --user -e ".[dev]"

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/home/test-project/.local/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r test-project && useradd -r -g test-project test-project

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/test-project/.local

# Copy application code
COPY --chown=test-project:test-project . .

# Install production package
RUN pip install --no-cache-dir --user -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache \
    && chown -R test-project:test-project /app

# Switch to non-root user
USER test-project

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir --user -e ".[dev]"

# Set environment variables for development
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=development \
    DEBUG=true

# Expose port
EXPOSE 8000

# Default command for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
