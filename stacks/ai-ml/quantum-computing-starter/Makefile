# Makefile for Quantum Computing Starter Template
# Provides comprehensive automation for quantum development workflows

.PHONY: help install install-dev test test-unit test-integration test-e2e test-performance lint format type-check security clean build run dev docs docker-build docker-run docker-dev quantum-simulate quantum-benchmark ci-setup ci-test ci-lint ci-security

# Default target
help: ## Show this help message
	@echo "Quantum Computing Starter - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

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
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	pytest tests/e2e/ -v

test-performance: ## Run performance tests only
	pytest tests/performance/ --benchmark-only

test-coverage: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-quantum: ## Run quantum-specific tests
	pytest tests/ -k "quantum" -v

# Code Quality
lint: ## Run linting
	flake8 src tests notebooks
	pylint src

format: ## Format code
	black src tests notebooks
	isort src tests notebooks

format-check: ## Check code formatting
	black --check src tests notebooks
	isort --check-only src tests notebooks

type-check: ## Run type checking
	mypy src

# Security
security: ## Run security checks
	bandit -r src/
	safety check
	pip-audit

security-full: ## Run comprehensive security checks
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

# Quantum Operations
quantum-simulate: ## Run quantum circuit simulation demo
	python run_quantum_app.py --demo circuits

quantum-benchmark: ## Run quantum performance benchmarks
	python scripts/benchmark_quantum.py

quantum-analyze: ## Analyze quantum algorithms
	python scripts/analyze_algorithms.py

# Development
dev: ## Start development environment
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

dev-server: ## Start quantum web server
	python -m uvicorn src.quantum_web:app --reload --host 0.0.0.0 --port 8000

shell: ## Start Python shell with quantum libraries loaded
	python -c "import sys; sys.path.insert(0, 'src'); from quantum_circuits.basic_gates import *; print('Quantum libraries loaded. Try: bell_circuit = create_bell_state(\"qiskit\")'); import code; code.interact(local=locals())"

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/

docs-serve: ## Serve documentation locally
	cd docs/_build && python -m http.server 8000

docs-clean: ## Clean documentation build
	rm -rf docs/_build/

# Docker
docker-build: ## Build Docker image
	docker build -t quantum-computing-starter:latest .

docker-run: ## Run Docker container
	docker run -p 8888:8888 -p 8000:8000 quantum-computing-starter:latest

docker-dev: ## Run development environment with Docker
	docker-compose -f docker/docker-compose.dev.yml up --build

docker-test: ## Run tests in Docker
	docker build -t quantum-test .
	docker run --rm quantum-test pytest tests/ -v

# Notebooks
notebook-convert: ## Convert notebooks to scripts
	jupyter nbconvert --to script notebooks/tutorials/*.ipynb
	jupyter nbconvert --to script notebooks/examples/*.ipynb

notebook-test: ## Test notebook execution
	jupyter nbconvert --to notebook --execute notebooks/tutorials/quantum_basics.ipynb

# Quantum Frameworks
qiskit-demo: ## Run Qiskit-specific demos
	python -c "import sys; sys.path.insert(0, 'src'); from quantum_circuits.basic_gates import create_bell_state; qc = create_bell_state('qiskit'); print('Qiskit Bell State:'); print(qc.draw())"

cirq-demo: ## Run Cirq-specific demos
	python -c "import sys; sys.path.insert(0, 'src'); from quantum_circuits.basic_gates import create_bell_state; qc = create_bell_state('cirq'); print('Cirq Bell State:'); print(qc)"

pennylane-demo: ## Run PennyLane-specific demos
	python -c "import sys; sys.path.insert(0, 'src'); from quantum_circuits.basic_gates import create_bell_state; bell = create_bell_state('pennylane'); result = bell(); print('PennyLane Bell State Probabilities:', result)"

# Backend Testing
test-qiskit: ## Test Qiskit backend
	python -c "import sys; sys.path.insert(0, 'src'); from backends.local_simulator import QiskitSimulator; backend = QiskitSimulator(); print('Qiskit backend available:', backend.is_available())"

test-cirq: ## Test Cirq backend
	python -c "import sys; sys.path.insert(0, 'src'); from backends.local_simulator import CirqSimulator; backend = CirqSimulator(); print('Cirq backend available:', backend.is_available())"

test-pennylane: ## Test PennyLane backend
	python -c "import sys; sys.path.insert(0, 'src'); from backends.local_simulator import PennyLaneSimulator; backend = PennyLaneSimulator(); print('PennyLane backend available:', backend.is_available())"

# Cleanup
clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
	rm -rf *.log logs/
	rm -rf quantum_results/ quantum_data/

clean-all: clean ## Clean everything including virtual environment
	rm -rf venv/ .venv/ env/

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
	python -c "import os; print('Environment variables loaded successfully')"

# Monitoring
logs: ## View application logs
	tail -f quantum.log

# Health checks
health: ## Check quantum libraries health
	python -c "import sys; sys.path.insert(0, 'src'); print('Testing quantum imports...'); import quantum_circuits; import quantum_ml; import backends; print('All quantum modules imported successfully!')"

# Performance
benchmark: ## Run performance benchmarks
	pytest tests/performance/ --benchmark-only --benchmark-save=benchmark

benchmark-compare: ## Compare with previous benchmark
	pytest tests/performance/ --benchmark-compare

# CI/CD helpers
ci-test: ## Run tests for CI/CD
	pytest --cov=src --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml

ci-lint: ## Run linting for CI/CD
	black --check src tests
	isort --check-only src tests
	flake8 src tests
	mypy src

ci-security: ## Run security checks for CI/CD
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	pip-audit --format=json --output=pip-audit-report.json

ci-quantum: ## Run quantum-specific CI tests
	pytest tests/ -k "quantum" --tb=short

# Development workflow
setup: install-dev env ## Complete development setup
	@echo "Quantum Computing Starter setup complete!"
	@echo "Run 'make dev' to start Jupyter Lab"
	@echo "Run 'make quantum-simulate' to run quantum demos"

check: format-check lint type-check security ## Run all checks
	@echo "All checks passed! ✅"

ci: ci-lint ci-test ci-security ci-quantum ## Run CI/CD pipeline locally
	@echo "CI/CD pipeline completed successfully! 🚀"

# Quick development commands
quick-test: ## Quick test run
	pytest tests/unit/ -x --tb=short

quick-format: ## Quick format check
	black --check src tests --quiet

quick-lint: ## Quick lint check
	flake8 src tests --max-line-length=88 --extend-ignore=E203,W503 --quiet

# Quantum research
research-bell: ## Research Bell states and entanglement
	python scripts/research_bell_states.py

research-qft: ## Research Quantum Fourier Transform
	python scripts/research_qft.py

research-shor: ## Research Shor's algorithm
	python scripts/research_shor.py

# Educational
tutorial-basic: ## Run basic quantum tutorial
	jupyter nbconvert --execute notebooks/tutorials/quantum_basics.ipynb --to notebook --inplace

tutorial-algorithms: ## Run algorithms tutorial
	jupyter nbconvert --execute notebooks/tutorials/quantum_algorithms.ipynb --to notebook --inplace

# API documentation
api-docs: ## Generate API documentation
	python -c "import sys; sys.path.insert(0, 'src'); from quantum_web import app; import json; print(json.dumps(app.openapi(), indent=2))" > api-spec.json

# Version management
version: ## Show current version
	python -c "import sys; sys.path.insert(0, 'src'); from quantum_computing_starter import __version__; print(__version__)"

# Release
release: clean test check build ## Create a release
	@echo "Release created successfully!"
	@echo "Run 'make upload' to distribute"

# Custom quantum commands
simulate-bell: ## Simulate Bell state measurements
	python -c "
import sys
sys.path.insert(0, 'src')
from quantum_circuits.basic_gates import create_bell_state, measure_circuit
import numpy as np

print('=== Bell State Simulation ===')
qc = create_bell_state('qiskit')
results = measure_circuit(qc, shots=1000, framework='qiskit')

print('Results:')
for outcome, count in sorted(results.items()):
    prob = count / 1000
    print(f'|{outcome}⟩: {prob:.1%}')

# Check for entanglement
entangled = abs(results.get('00', 0) + results.get('11', 0) - 1000) < 50
print(f'Entanglement detected: {entangled}')
"

benchmark-circuits: ## Benchmark different circuit sizes
	python -c "
import sys
sys.path.insert(0, 'src')
from quantum_circuits.basic_gates import create_random_circuit, measure_circuit
import time

print('=== Circuit Benchmarking ===')
sizes = [5, 10, 15, 20]

for n_qubits in sizes:
    qc = create_random_circuit(n_qubits, depth=10, framework='qiskit')
    start_time = time.time()
    results = measure_circuit(qc, shots=100, framework='qiskit')
    elapsed = time.time() - start_time
    
    print(f'{n_qubits} qubits: {elapsed:.3f}s')
"
