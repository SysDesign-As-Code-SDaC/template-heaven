"""
Unit tests for base benchmark framework.

This module tests the core benchmark infrastructure, including:
- Benchmark registration and discovery
- Base benchmark execution lifecycle
- Result validation and formatting
- Error handling and edge cases
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from ai_benchmarks.core.base_benchmark import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRegistry,
    benchmark_registry
)


class MockBenchmark(BaseBenchmark):
    """Mock benchmark implementation for testing."""

    CATEGORY = "test"
    NAME = "mock_benchmark"

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.setup_called = False
        self.run_called = False
        self.cleanup_called = False

    def setup(self, **kwargs):
        self.setup_called = True

    def run(self, model, **kwargs):
        self.run_called = True
        return BenchmarkResult(
            benchmark_name=self.config.name,
            benchmark_category=self.config.category,
            model_name=getattr(model, 'name', 'test_model'),
            score=0.85,
            metrics={'test_metric': 42},
            metadata={'test_info': 'mock_result'}
        )

    def validate_result(self, result):
        return isinstance(result, BenchmarkResult) and 0 <= result.score <= 1

    def cleanup(self):
        self.cleanup_called = True


class FailingBenchmark(MockBenchmark):
    """Mock benchmark that fails during execution."""

    def run(self, model, **kwargs):
        raise RuntimeError("Simulated benchmark failure")


@pytest.fixture
def sample_config():
    """Sample benchmark configuration for testing."""
    return BenchmarkConfig(
        name="test_benchmark",
        category="test",
        description="Test benchmark for unit testing",
        parameters={"test_param": "value"},
        timeout=60.0
    )


@pytest.fixture
def mock_model():
    """Mock AI model for testing."""
    model = Mock()
    model.name = "test_model"
    return model


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_config_creation(self):
        """Test benchmark configuration creation."""
        config = BenchmarkConfig(
            name="test_config",
            category="test",
            description="Test configuration"
        )

        assert config.name == "test_config"
        assert config.category == "test"
        assert config.description == "Test configuration"
        assert config.parameters == {}
        assert config.timeout is None

    def test_config_with_parameters(self):
        """Test configuration with custom parameters."""
        params = {"learning_rate": 0.001, "epochs": 100}
        config = BenchmarkConfig(
            name="param_test",
            category="test",
            description="Parameter test",
            parameters=params,
            timeout=300.0
        )

        assert config.parameters == params
        assert config.timeout == 300.0


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_result_creation(self):
        """Test benchmark result creation."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            benchmark_category="test",
            model_name="test_model",
            score=0.85
        )

        assert result.benchmark_name == "test_benchmark"
        assert result.benchmark_category == "test"
        assert result.model_name == "test_model"
        assert result.score == 0.85
        assert isinstance(result.timestamp, datetime)
        assert result.status == "completed"

    def test_result_to_dict(self):
        """Test result serialization to dictionary."""
        result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="test",
            model_name="model",
            score=0.75,
            metrics={"accuracy": 0.8},
            metadata={"version": "1.0"}
        )

        result_dict = result.to_dict()

        assert result_dict["benchmark_name"] == "test"
        assert result_dict["score"] == 0.75
        assert result_dict["metrics"]["accuracy"] == 0.8
        assert "timestamp" in result_dict

    def test_result_to_json(self):
        """Test result serialization to JSON."""
        result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="test",
            model_name="model",
            score=0.9
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)

        # Should be parseable back
        import json
        parsed = json.loads(json_str)
        assert parsed["score"] == 0.9


class TestBenchmarkRegistry:
    """Test benchmark registry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        # Clear the registry for clean testing
        BenchmarkRegistry._benchmarks.clear()

    def teardown_method(self):
        """Clean up after each test."""
        BenchmarkRegistry._benchmarks.clear()

    def test_register_benchmark(self):
        """Test benchmark registration."""
        BenchmarkRegistry.register("test_category", "test_benchmark", MockBenchmark)

        assert "test_category" in BenchmarkRegistry._benchmarks
        assert "test_benchmark" in BenchmarkRegistry._benchmarks["test_category"]
        assert BenchmarkRegistry._benchmarks["test_category"]["test_benchmark"] == MockBenchmark

    def test_get_registered_benchmark(self):
        """Test retrieving registered benchmarks."""
        BenchmarkRegistry.register("test", "mock", MockBenchmark)

        retrieved = BenchmarkRegistry.get_benchmark("test", "mock")
        assert retrieved == MockBenchmark

    def test_get_nonexistent_benchmark(self):
        """Test retrieving non-existent benchmark."""
        result = BenchmarkRegistry.get_benchmark("nonexistent", "benchmark")
        assert result is None

    def test_list_categories(self):
        """Test listing all categories."""
        BenchmarkRegistry.register("cat1", "bench1", MockBenchmark)
        BenchmarkRegistry.register("cat2", "bench2", MockBenchmark)

        categories = BenchmarkRegistry.list_categories()
        assert "cat1" in categories
        assert "cat2" in categories

    def test_list_benchmarks_in_category(self):
        """Test listing benchmarks in a specific category."""
        BenchmarkRegistry.register("test", "bench1", MockBenchmark)
        BenchmarkRegistry.register("test", "bench2", MockBenchmark)
        BenchmarkRegistry.register("other", "bench3", MockBenchmark)

        benchmarks = BenchmarkRegistry.list_benchmarks("test")
        assert "test" in benchmarks
        assert "bench1" in benchmarks["test"]
        assert "bench2" in benchmarks["test"]
        assert "bench3" not in benchmarks["test"]

    def test_list_all_benchmarks(self):
        """Test listing all benchmarks across categories."""
        BenchmarkRegistry.register("cat1", "bench1", MockBenchmark)
        BenchmarkRegistry.register("cat2", "bench2", MockBenchmark)

        all_benchmarks = BenchmarkRegistry.list_benchmarks()
        assert "cat1" in all_benchmarks
        assert "cat2" in all_benchmarks
        assert "bench1" in all_benchmarks["cat1"]


class TestBaseBenchmark:
    """Test base benchmark functionality."""

    @pytest.fixture
    def benchmark(self, sample_config):
        """Create a mock benchmark instance."""
        return MockBenchmark(sample_config)

    def test_benchmark_initialization(self, benchmark, sample_config):
        """Test benchmark initialization."""
        assert benchmark.config == sample_config
        assert hasattr(benchmark, 'logger')
        assert benchmark.setup_called is False
        assert benchmark.run_called is False
        assert benchmark.cleanup_called is False

    @pytest.mark.asyncio
    async def test_successful_execution(self, benchmark, mock_model):
        """Test successful benchmark execution."""
        result = await benchmark.execute_async(mock_model)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "test_benchmark"
        assert result.benchmark_category == "test"
        assert result.model_name == "test_model"
        assert result.score == 0.85
        assert result.status == "completed"
        assert benchmark.setup_called is True
        assert benchmark.run_called is True
        assert benchmark.cleanup_called is True

    @pytest.mark.asyncio
    async def test_execution_with_timing(self, benchmark, mock_model):
        """Test that execution time is recorded."""
        start_time = datetime.utcnow()
        result = await benchmark.execute_async(mock_model)
        end_time = datetime.utcnow()

        assert result.execution_time >= 0
        assert result.execution_time <= (end_time - start_time).total_seconds()

    def test_validate_valid_result(self, benchmark):
        """Test validation of valid results."""
        valid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="test",
            model_name="model",
            score=0.75
        )

        assert benchmark.validate_result(valid_result) is True

    def test_validate_invalid_result(self, benchmark):
        """Test validation of invalid results."""
        # Invalid score
        invalid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="test",
            model_name="model",
            score=1.5  # Invalid score > 1
        )

        assert benchmark.validate_result(invalid_result) is False

    @pytest.mark.asyncio
    async def test_execution_error_handling(self, mock_model):
        """Test error handling during execution."""
        config = BenchmarkConfig(
            name="failing_benchmark",
            category="test",
            description="Failing benchmark for testing"
        )

        failing_benchmark = FailingBenchmark(config)

        result = await failing_benchmark.execute_async(mock_model)

        assert isinstance(result, BenchmarkResult)
        assert result.status == "failed"
        assert result.score == 0.0
        assert "error" in result.metadata
        assert failing_benchmark.cleanup_called is True


class TestBenchmarkDecorator:
    """Test benchmark registry decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        BenchmarkRegistry._benchmarks.clear()

    def teardown_method(self):
        """Clean up after each test."""
        BenchmarkRegistry._benchmarks.clear()

    def test_decorator_registration(self):
        """Test that decorator automatically registers benchmarks."""

        @benchmark_registry
        class DecoratedBenchmark(MockBenchmark):
            CATEGORY = "decorated"
            NAME = "auto_registered"

        # Check that it was registered
        registered = BenchmarkRegistry.get_benchmark("decorated", "auto_registered")
        assert registered == DecoratedBenchmark

    def test_decorator_without_attributes(self):
        """Test decorator on class without required attributes."""

        @benchmark_registry
        class IncompleteBenchmark(BaseBenchmark):
            # Missing CATEGORY and NAME attributes
            pass

        # Should not register classes without proper attributes
        registered = BenchmarkRegistry.get_benchmark("incomplete", "benchmark")
        assert registered is None


class TestBenchmarkExecutionLifecycle:
    """Test complete benchmark execution lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, sample_config, mock_model):
        """Test complete benchmark execution lifecycle."""
        benchmark = MockBenchmark(sample_config)

        # Execute benchmark
        result = await benchmark.execute_async(mock_model)

        # Verify all lifecycle methods were called
        assert benchmark.setup_called
        assert benchmark.run_called
        assert benchmark.cleanup_called

        # Verify result structure
        assert result.benchmark_name == sample_config.name
        assert result.benchmark_category == sample_config.category
        assert result.model_name == mock_model.name
        assert isinstance(result.score, (int, float))
        assert 0 <= result.score <= 1

    @pytest.mark.asyncio
    async def test_lifecycle_with_exceptions(self, mock_model):
        """Test lifecycle when exceptions occur."""
        config = BenchmarkConfig(
            name="exception_test",
            category="test",
            description="Test exception handling"
        )

        # Create benchmark that fails in setup
        class SetupFailingBenchmark(MockBenchmark):
            def setup(self, **kwargs):
                raise ValueError("Setup failed")

        benchmark = SetupFailingBenchmark(config)
        result = await benchmark.execute_async(mock_model)

        # Should still call cleanup even if setup fails
        assert benchmark.cleanup_called
        assert result.status == "failed"
        assert "error" in result.metadata
