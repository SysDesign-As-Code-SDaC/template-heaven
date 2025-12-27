"""
Comprehensive Pytest Testing Template - Summary

This template provides a complete testing setup using pytest with comprehensive
features for modern Python development.

## Features Included

### Core Testing Framework
- **pytest**: Main testing framework with rich assertions and fixtures
- **pytest-cov**: Code coverage reporting
- **pytest-xdist**: Parallel test execution
- **pytest-mock**: Enhanced mocking capabilities
- **pytest-asyncio**: Async testing support
- **faker**: Generate fake test data
- **factory-boy**: Create test object factories

### Test Organization
- **Unit Tests** (`tests/test_unit/`): Test individual functions and classes
- **Integration Tests** (`tests/test_integration/`): Test component interactions
- **End-to-End Tests** (`tests/test_e2e/`): Test complete user journeys
- **Performance Tests** (`tests/performance/`): Test speed and resource usage

### Test Infrastructure
- **conftest.py**: Global fixtures and pytest configuration
- **test_helpers.py**: Shared utilities and test data factories
- **pytest.ini**: Test configuration and settings

### Test Categories & Markers
- `@pytest.mark.slow`: Mark slow-running tests
- `@pytest.mark.integration`: Mark integration tests
- `@pytest.mark.e2e`: Mark end-to-end tests
- `@pytest.mark.performance`: Mark performance tests
- `@pytest.mark.database`: Mark database-dependent tests
- `@pytest.mark.api`: Mark API tests

### Fixtures Provided
- **Database**: Mock and real database sessions
- **HTTP**: Mock HTTP clients and responses
- **Caching**: Redis and in-memory cache mocks
- **External APIs**: Mock external service responses
- **Authentication**: Mock auth tokens and sessions
- **File System**: Temporary files and directories
- **Async**: Async database sessions and HTTP clients

### Test Data Generation
- **Users**: Complete user profiles with preferences
- **Companies**: Business entities with contact info
- **Products**: E-commerce products with pricing
- **Orders**: Complete order structures with items
- **API Responses**: Mock HTTP responses with proper status codes

### Performance Testing
- **Benchmarking**: Measure execution times and throughput
- **Memory Monitoring**: Track memory usage during tests
- **Load Testing**: Simulate concurrent user scenarios
- **Scalability Testing**: Test horizontal scaling characteristics

### Code Quality Integration
- **Coverage Reporting**: HTML and terminal coverage reports
- **Parallel Execution**: Run tests across multiple CPU cores
- **Test Selection**: Run specific test categories or markers
- **Automation Ready**: Pre-configured for automated testing pipelines

## Usage Examples

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "integration and not slow"
pytest -m "performance"

# Run in parallel
pytest -n auto

# Run tests for specific file
pytest tests/test_unit/test_core_logic.py
```

### Writing Tests
```python
import pytest
from tests.helpers.test_helpers import test_data_factory, assertion_helpers

def test_user_creation(fake_user, mock_db_session):
    # Arrange
    user_data = test_data_factory.create_user({"name": "Test User"})

    # Act
    # result = user_service.create_user(user_data)

    # Assert
    # assertion_helpers.assert_dict_contains_subset(user_data, result)

def test_api_endpoint(async_client):
    # Arrange
    test_data = {"name": "Test Item"}

    # Act
    # response = await async_client.post("/api/items", json=test_data)

    # Assert
    # assertion_helpers.assert_response_status(response, 201)
```

### Performance Testing
```python
@pytest.mark.performance
def test_data_processing_performance(benchmark, faker):
    # Arrange
    large_dataset = [faker.text() for _ in range(1000)]

    # Act & Assert
    result = benchmark(process_data, large_dataset)
    assert benchmark.stats["mean"] < 1.0  # Under 1 second
```

## Configuration Files

### pytest.ini
- Test discovery patterns
- Custom markers
- Coverage settings
- Parallel execution configuration

### conftest.py
- Global fixtures for all test modules
- Pytest configuration and hooks
- Shared test utilities

### requirements-dev.txt
- All testing dependencies
- Development tools
- Code quality checkers

## Best Practices Implemented

1. **Test Isolation**: Each test is independent and can run in any order
2. **Fixture Management**: Comprehensive fixtures for common test scenarios
3. **Mocking Strategy**: Proper use of mocks for external dependencies
4. **Data Management**: Fake data generation for realistic test scenarios
5. **Performance Monitoring**: Built-in performance testing capabilities
- 6. **Automation Integration**: Ready for automated testing pipelines
7. **Code Coverage**: Comprehensive coverage reporting and analysis
8. **Parallel Execution**: Optimized for fast test execution
9. **Error Handling**: Proper error testing and exception handling
10. **Documentation**: Extensive documentation and examples

## Integration with Development Workflow

This template integrates seamlessly with modern Python development practices:

- **TDD/BDD**: Supports test-driven and behavior-driven development
- **Automation examples**: Pre-configured for GitHub Actions, GitLab CI, and other platforms (examples disabled)
- **Code Quality**: Compatible with black, isort, flake8, mypy, and other tools
- **Documentation**: Generates test reports and coverage documentation
- **Debugging**: Rich test output with detailed failure information
- **Profiling**: Performance profiling and bottleneck identification

## Customization

The template is highly customizable:

- Add new fixtures in `conftest.py`
- Create custom test data factories
- Add new test markers and categories
- Configure coverage settings
- Add new performance test utilities
- Integrate with your specific database/API stack

This comprehensive testing template provides everything needed for robust,
maintainable, and scalable test suites in modern Python applications.
"""