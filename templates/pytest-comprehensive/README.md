# pytest-comprehensive

A comprehensive pytest testing framework template with fixtures, mocking, coverage, parallel testing, and automation integration (CI examples disabled).

## Features

- **Comprehensive Test Fixtures**: Pre-configured fixtures for common testing scenarios
- **Mocking & Stubbing**: Advanced mocking with pytest-mock and responses
- **Coverage Reporting**: Detailed coverage analysis with multiple output formats
- **Parallel Execution**: Speed up tests with pytest-xdist
- **Async Testing**: Support for async/await test functions
- **CI Integration (example)**: GitHub Actions workflow (disabled)
- **Test Data Generation**: Faker integration for realistic test data
- **Factory Pattern**: Factory Boy for complex object creation
- **Property-based Testing**: Hypothesis integration for advanced testing
- **Performance Testing**: Basic performance benchmarking
- **Test Organization**: Well-structured test directory with examples

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run in parallel
pytest -n auto

# Run specific test file
pytest tests/test_example.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_user"
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_unit/              # Unit tests
├── test_integration/       # Integration tests
├── test_e2e/              # End-to-end tests
├── fixtures/              # Test data and fixtures
├── helpers/               # Test helper functions
└── performance/           # Performance tests
```

## Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    performance: Performance tests
```

### Coverage Configuration

```ini
[coverage:run]
source = src
omit =
    */tests/*
    */venv/*
    */env/*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
```

## Fixtures

### Built-in Fixtures

- `temp_dir`: Temporary directory for file operations
- `sample_user`: Sample user object for testing
- `mock_api_client`: Mocked API client
- `database_session`: Database session for integration tests
- `faker`: Faker instance for generating test data

### Custom Fixtures

```python
@pytest.fixture
def authenticated_client(client, sample_user):
    """Client authenticated with sample user"""
    client.force_authenticate(user=sample_user)
    return client

@pytest.fixture
def api_response():
    """Mock API response"""
    return {
        "id": 1,
        "name": "Test Item",
        "created_at": "2024-01-01T00:00:00Z"
    }
```

## Mocking Examples

### Function Mocking

```python
def test_user_creation_with_mock(mocker):
    # Mock external API call
    mock_api = mocker.patch('myapp.api.external_api_call')
    mock_api.return_value = {"status": "success"}

    result = create_user({"name": "John"})
    assert result["status"] == "success"
    mock_api.assert_called_once()
```

### HTTP Request Mocking

```python
def test_api_integration(requests_mock):
    # Mock HTTP requests
    requests_mock.get(
        'https://api.example.com/users',
        json=[{"id": 1, "name": "John"}]
    )

    users = get_users_from_api()
    assert len(users) == 1
    assert users[0]["name"] == "John"
```

## Async Testing

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == "expected_value"

@pytest.mark.asyncio
async def test_async_with_mock(mocker):
    mock_func = mocker.patch('myapp.async_utils.async_call')
    mock_func.return_value = "mocked"

    result = await process_async_data()
    assert result == "mocked"
```

## Property-based Testing

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_string_processing(input_text):
    result = process_string(input_text)
    assert isinstance(result, str)
    assert len(result) > 0

@given(st.lists(st.integers(), min_size=1, max_size=10))
def test_list_operations(numbers):
    result = sum_positive_numbers(numbers)
    assert result >= 0
```

## Performance Testing

```python
def test_performance_baseline(benchmark):
    @benchmark
    def run():
        result = expensive_operation()
        return result

def test_memory_usage():
    import tracemalloc
    tracemalloc.start()

    perform_operation()

    current, peak = tracemalloc.get_traced_memory()
    assert peak < 100 * 1024 * 1024  # Less than 100MB
```

## Automation Integration

### GitHub Actions (disabled)

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Best Practices

### Test Organization

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete user workflows
4. **Performance Tests**: Monitor performance regressions

### Naming Conventions

- `test_function_name`: Unit test for specific function
- `test_class_name`: Test class for related functionality
- `TestClassName`: Test class following class naming
- `test_feature_scenario`: BDD-style test names

### Test Data Management

- Use factories for complex object creation
- Faker for realistic test data
- Fixtures for reusable test setup
- Separate test data from production data

### Assertions

- Use descriptive assertion messages
- Test both positive and negative cases
- Verify side effects and state changes
- Use appropriate assertion methods

## Advanced Features

### Custom Markers

```python
@pytest.mark.slow
def test_slow_operation():
    # Test that takes a long time
    pass

@pytest.mark.integration
def test_database_integration():
    # Integration test requiring database
    pass
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```

### Conditional Testing

```python
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9+")
def test_python39_feature():
    # Test Python 3.9+ specific features
    pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure proper Python path configuration
2. **Fixture Errors**: Check fixture dependencies and scope
3. **Coverage Issues**: Verify source paths and omit patterns
4. **Async Issues**: Use proper async test decorators

### Debugging Tests

```bash
# Run with debugging
pytest --pdb

# Run specific failing test
pytest tests/test_failing.py::TestClass::test_method -v

# Show fixtures for test
pytest --fixtures tests/test_example.py
```

## Contributing

1. Write tests for new features
2. Maintain test coverage above 80%
3. Follow existing test patterns
4. Update documentation for test changes
5. Run full test suite before submitting PR

## License

This template is licensed under the MIT License.