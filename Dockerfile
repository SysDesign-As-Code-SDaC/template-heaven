# Template Heaven - Multi-stage Docker Build
# This Dockerfile follows security best practices and uses multi-stage builds
# for optimal image size and security

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
    PATH="/home/templateheaven/.local/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r templateheaven && \
    useradd -r -g templateheaven -d /home/templateheaven -s /bin/bash templateheaven

# Create application directory
RUN mkdir -p /app /home/templateheaven/.templateheaven && \
    chown -R templateheaven:templateheaven /app /home/templateheaven

# Switch to non-root user
USER templateheaven

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder --chown=templateheaven:templateheaven /root/.local /home/templateheaven/.local

# Copy application code
COPY --chown=templateheaven:templateheaven . .

# Install package
RUN pip install --no-cache-dir -e .

# Create cache directory
RUN mkdir -p /home/templateheaven/.templateheaven/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import templateheaven; print('OK')" || exit 1

# Default command
ENTRYPOINT ["templateheaven"]
CMD ["--help"]

# Security: Run as non-root user
USER templateheaven

# Expose port (if needed for future web UI)
EXPOSE 8080

# Labels for metadata
LABEL maintainer="Template Heaven Team" \
      version="0.1.0" \
      description="Interactive template management for modern software development" \
      org.opencontainers.image.title="Template Heaven" \
      org.opencontainers.image.description="Interactive template management for modern software development" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="Template Heaven Team" \
      org.opencontainers.image.url="https://github.com/template-heaven/templateheaven" \
      org.opencontainers.image.source="https://github.com/template-heaven/templateheaven" \
      org.opencontainers.image.licenses="MIT"
