# AI Benchmarks Suite - Test Suite

Comprehensive test suite for the AI Benchmarks Suite, covering all benchmark categories and components.

## 🧪 Test Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── test_base_benchmark.py     # Base benchmark framework tests
│   ├── test_asi_benchmarks.py     # ASI benchmark tests
│   ├── test_agi_benchmarks.py     # AGI benchmark tests
│   ├── test_neuromorphic_benchmarks.py  # Neuromorphic benchmark tests
│   └── test_hybrid_llm_benchmarks.py    # Hybrid LLM benchmark tests
├── integration/                   # Integration tests
│   ├── test_benchmark_runner.py   # Benchmark runner integration
│   └── test_cli_integration.py    # CLI integration tests
├── performance/                   # Performance tests
├── validation/                    # Validation tests
├── cli/                          # CLI-specific tests
├── conftest.py                   # Test configuration and fixtures
├── __init__.py
└── README.md                     # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
# From the ai-benchmarks-suite directory
pytest tests/

# With coverage report
pytest --cov=ai_benchmarks --cov-report=html

# Verbose output
pytest -v tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_base_benchmark.py

# Specific test function
pytest tests/unit/test_base_benchmark.py::TestBenchmarkRegistry::test_register_benchmark
```

### Run Tests with Different Configurations
```bash
# Run tests in parallel (if available)
pytest -n auto tests/

# Run with different Python versions (if using tox)
tox

# Run tests matching a pattern
pytest -k "test_benchmark" tests/

# Run tests for a specific benchmark category
pytest -k "asi" tests/
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

**test_base_benchmark.py**
- Benchmark configuration validation
- Benchmark registry functionality
- Base benchmark lifecycle (setup, run, cleanup)
- Result validation and serialization
- Error handling and edge cases

**test_asi_benchmarks.py**
- Recursive self-improvement benchmark
- Universal problem solver benchmark
- Safety monitoring functionality
- Performance metrics calculation

**test_agi_benchmarks.py**
- General intelligence benchmark
- Cognitive domain evaluation
- Transfer learning assessment
- Knowledge integration testing

**test_neuromorphic_benchmarks.py**
- Spiking neural network benchmarks
- Energy monitoring and efficiency
- Temporal processing evaluation
- Neural plasticity assessment

**test_hybrid_llm_benchmarks.py**
- Hybrid architecture evaluation
- Ensemble coordination testing
- Multi-modal integration
- Failure simulation and robustness

### Integration Tests (`tests/integration/`)

**test_benchmark_runner.py**
- Complete benchmark suite execution
- Parallel benchmark processing
- Result aggregation and reporting
- Error handling across benchmarks
- Configuration loading and validation

**test_cli_integration.py**
- Command-line argument parsing
- CLI workflow execution
- Output formatting and reporting
- Error handling in CLI context

## 📊 Test Fixtures

### Common Fixtures (`conftest.py`)
- `sample_config`: Standard benchmark configuration
- `mock_model`: Mock AI model for testing
- `temp_output_dir`: Temporary directory for test outputs

### Benchmark-Specific Fixtures
- `asi_config`: ASI benchmark configuration
- `agi_config`: AGI benchmark configuration
- `neuromorphic_config`: Neuromorphic benchmark configuration
- `hybrid_config`: Hybrid LLM benchmark configuration

## 🐛 Debugging Tests

### Running Tests with Debug Output
```bash
# Enable debug logging
pytest -s -v --log-cli-level=DEBUG tests/unit/test_base_benchmark.py

# Run specific test with pdb
pytest --pdb tests/unit/test_base_benchmark.py::TestBenchmarkRegistry::test_register_benchmark
```

### Common Test Issues

**Import Errors**
```bash
# Ensure the package is properly installed
pip install -e .

# Or run tests from the correct directory
cd ai-benchmarks-suite
pytest tests/
```

**Async Test Issues**
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Run async tests specifically
pytest -k "async" tests/
```

**Coverage Issues**
```bash
# Generate coverage report
pytest --cov=ai_benchmarks --cov-report=html

# Open coverage report in browser
open htmlcov/index.html
```

## 📈 Test Coverage

### Coverage Goals
- **Unit Tests**: >90% coverage of core functionality
- **Integration Tests**: Complete workflow coverage
- **Error Handling**: Comprehensive error scenario testing

### Coverage Report
```bash
pytest --cov=ai_benchmarks --cov-report=term-missing
```

## 🛠️ Writing New Tests

### Adding Unit Tests
```python
import pytest
from ai_benchmarks.core.base_benchmark import BenchmarkConfig

def test_new_functionality():
    """Test description."""
    config = BenchmarkConfig(
        name="test",
        category="test",
        description="Test config"
    )

    # Test implementation
    assert config.name == "test"
```

### Adding Integration Tests
```python
import pytest
from ai_benchmarks import BenchmarkRunner, create_quick_evaluation_suite

@pytest.mark.asyncio
async def test_integration_workflow(mock_model, temp_output_dir):
    """Test complete workflow integration."""
    config = create_quick_evaluation_suite()
    config.output_directory = str(temp_output_dir)

    runner = BenchmarkRunner(config)
    results = await runner.run_benchmark_suite(mock_model)

    assert results.total_benchmarks > 0
    assert results.successful_benchmarks >= 0
```

### Test Organization Guidelines
- **One test per behavior**: Each test should verify one specific behavior
- **Descriptive names**: Test names should clearly describe what they're testing
- **Independent tests**: Tests should not depend on each other
- **Fast execution**: Tests should run quickly to enable frequent execution
- **Use fixtures**: Reuse common test setup with pytest fixtures

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    - name: Run tests
      run: pytest --cov=ai_benchmarks --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        args: [--cov=ai_benchmarks, --cov-report=term-missing]
```

## 📋 Test Checklist

### Before Committing
- [ ] All tests pass: `pytest tests/`
- [ ] Coverage meets requirements: `pytest --cov=ai_benchmarks`
- [ ] No linting errors: `flake8 ai_benchmarks tests/`
- [ ] Type checking passes: `mypy ai_benchmarks`

### Before Merging
- [ ] Integration tests pass in CI/CD
- [ ] Performance tests meet benchmarks
- [ ] Documentation updated for new functionality
- [ ] Changelog updated with changes

## 🐛 Troubleshooting

### Common Issues

**Tests Hanging**
```bash
# Check for async test issues
pytest --tb=short -x tests/

# Run with timeout
pytest --timeout=300 tests/
```

**Import Errors in Tests**
```bash
# Ensure test directory structure
python -c "import sys; print(sys.path)"

# Check package installation
pip show ai-benchmarks-suite
```

**Fixture Errors**
```bash
# List available fixtures
pytest --fixtures tests/

# Run specific fixture
pytest --fixture=session --fixture-per-session=tmp_path
```

## 📊 Test Metrics

### Key Metrics
- **Test Count**: Total number of test functions
- **Coverage Percentage**: Code coverage by tests
- **Execution Time**: Time to run full test suite
- **Failure Rate**: Percentage of failing tests

### Monitoring
```bash
# Generate test metrics
pytest --durations=10 --cov=ai_benchmarks --cov-report=html tests/

# Benchmark test execution
pytest --benchmark-only tests/
```

## 🤝 Contributing

### Adding Tests for New Benchmarks
1. Create unit test file: `tests/unit/test_{category}_benchmarks.py`
2. Implement comprehensive test coverage
3. Add integration tests in `tests/integration/`
4. Update this README with new test information
5. Ensure all tests pass before submitting PR

### Test Best Practices
- **DRY Principle**: Don't Repeat Yourself - use fixtures and helper functions
- **Descriptive Assertions**: Use clear assertion messages
- **Parameterized Tests**: Use `@pytest.mark.parametrize` for similar test cases
- **Mock External Dependencies**: Use mocks for external APIs and services
- **Clean Test Data**: Use realistic but sanitized test data
