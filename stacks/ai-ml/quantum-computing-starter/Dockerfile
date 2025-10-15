# Quantum Computing Starter Dockerfile
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
    libgomp1 \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN groupadd -r quantum && useradd -r -g quantum quantum

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
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# =============================================================================
# Quantum frameworks stage - install heavy quantum packages separately
# =============================================================================
FROM dependencies as quantum-frameworks

# Install quantum computing frameworks
RUN pip install --no-cache-dir \
    qiskit \
    qiskit-aer \
    qiskit-ibmq-provider \
    qiskit-machine-learning \
    cirq \
    cirq-google \
    pennylane \
    pennylane-qiskit

# =============================================================================
# Development stage - full development environment
# =============================================================================
FROM quantum-frameworks as development

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipykernel \
    matplotlib \
    plotly \
    seaborn \
    notebook

# Setup Jupyter kernel
RUN python -m ipykernel install --user --name quantum-computing --display-name "Quantum Computing"

# Create directories
RUN mkdir -p /app/data /app/logs /app/notebooks && \
    chown -R quantum:quantum /app

# Copy application code
COPY . .

# Set permissions
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Expose ports
EXPOSE 8888 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from quantum_circuits.basic_gates import create_bell_state; print('Health check passed')" || exit 1

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Production stage - minimal runtime environment
# =============================================================================
FROM quantum-frameworks as production

# Install only runtime dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-dotenv \
    prometheus-client \
    structlog

# Create directories
RUN mkdir -p /app/data /app/logs && \
    chown -R quantum:quantum /app

# Copy application code (excluding development files)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./
COPY .env.example ./

# Set permissions
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.quantum_web:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Testing stage - for CI/CD pipelines
# =============================================================================
FROM quantum-frameworks as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-xdist \
    pytest-html \
    pytest-benchmark

# Create directories
RUN mkdir -p /app/test-results /app/coverage && \
    chown -R quantum:quantum /app

# Copy application code
COPY . .

# Set permissions
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Default command for testing
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]

# =============================================================================
# Benchmarking stage - for performance testing
# =============================================================================
FROM quantum-frameworks as benchmarking

# Install benchmarking tools
RUN pip install --no-cache-dir \
    pytest-benchmark \
    memory-profiler \
    line-profiler

# Create directories
RUN mkdir -p /app/benchmarks /app/profiles && \
    chown -R quantum:quantum /app

# Copy application code
COPY . .

# Set permissions
RUN chown -R quantum:quantum /app

# Switch to non-root user
USER quantum

# Default command for benchmarking
CMD ["python", "scripts/benchmark_quantum.py"]

# =============================================================================
# Default target - development environment
# =============================================================================
FROM development
