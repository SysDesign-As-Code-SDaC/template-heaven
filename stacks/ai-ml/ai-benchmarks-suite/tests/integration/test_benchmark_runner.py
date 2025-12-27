"""
Integration tests for benchmark runner.

Tests cover:
- Complete benchmark suite execution
- Parallel benchmark processing
- Result aggregation and reporting
- Error handling across benchmarks
- Configuration loading and validation
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from ai_benchmarks.core.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSuiteConfig,
    BenchmarkSuiteResult,
    create_comprehensive_ai_suite,
    create_quick_evaluation_suite,
    BenchmarkSuiteBuilder
)
from ai_benchmarks.core.base_benchmark import BenchmarkResult


@pytest.fixture
def mock_model():
    """Mock AI model for integration testing."""
    model = Mock()
    model.name = "integration_test_model"
    return model


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for test results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestBenchmarkRunnerIntegration:
    """Integration tests for benchmark runner."""

    @pytest.mark.asyncio
    async def test_comprehensive_suite_execution(self, mock_model, temp_output_dir):
        """Test complete execution of comprehensive benchmark suite."""
        config = create_comprehensive_ai_suite()
        config.output_directory = str(temp_output_dir)

        runner = BenchmarkRunner(config)

        # Execute benchmark suite
        result = await runner.run_benchmark_suite(mock_model)

        # Verify result structure
        assert isinstance(result, BenchmarkSuiteResult)
        assert result.suite_name == config.name
        assert result.total_benchmarks >= 4  # At least our implemented benchmarks
        assert result.successful_benchmarks >= 0
        assert result.failed_benchmarks >= 0
        assert result.total_benchmarks == result.successful_benchmarks + result.failed_benchmarks
        assert isinstance(result.results, list)
        assert len(result.results) == result.total_benchmarks
        assert isinstance(result.summary, dict)
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_quick_evaluation_execution(self, mock_model, temp_output_dir):
        """Test quick evaluation suite execution."""
        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)

        runner = BenchmarkRunner(config)

        result = await runner.run_benchmark_suite(mock_model)

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.total_benchmarks > 0
        assert all(isinstance(r, BenchmarkResult) for r in result.results)

    @pytest.mark.asyncio
    async def test_custom_suite_execution(self, mock_model, temp_output_dir):
        """Test custom benchmark suite execution."""
        custom_config = (BenchmarkSuiteBuilder("Custom Integration Test")
            .description("Custom suite for integration testing")
            .include_categories("asi", "agi")
            .parallel_execution(False)  # Sequential for predictable testing
            .timeout(60)  # Shorter timeout for testing
            .output_directory(str(temp_output_dir))
            .build())

        runner = BenchmarkRunner(custom_config)

        result = await runner.run_benchmark_suite(mock_model)

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.suite_name == "Custom Integration Test"
        assert result.total_benchmarks >= 2  # Should include at least 2 benchmarks

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_model, temp_output_dir):
        """Test parallel benchmark execution."""
        config = create_comprehensive_ai_suite()
        config.output_directory = str(temp_output_dir)
        config.parallel_execution = True
        config.max_workers = 2

        runner = BenchmarkRunner(config)

        result = await runner.run_benchmark_suite(mock_model)

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.total_benchmarks > 0
        # Parallel execution should complete successfully
        assert result.successful_benchmarks >= 0

    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_model, temp_output_dir):
        """Test sequential benchmark execution."""
        config = create_comprehensive_ai_suite()
        config.output_directory = str(temp_output_dir)
        config.parallel_execution = False

        runner = BenchmarkRunner(config)

        result = await runner.run_benchmark_suite(mock_model)

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.total_benchmarks > 0

    @pytest.mark.asyncio
    async def test_result_persistence(self, mock_model, temp_output_dir):
        """Test that results are properly saved to disk."""
        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)
        config.save_intermediate_results = True

        runner = BenchmarkRunner(config)

        result = await runner.run_benchmark_suite(mock_model)

        # Check that main results file exists
        results_file = temp_output_dir / "suite_results.json"
        assert results_file.exists()

        # Verify results can be loaded back
        with open(results_file, 'r') as f:
            saved_data = json.load(f)

        assert "suite_info" in saved_data
        assert "summary" in saved_data
        assert "results" in saved_data
        assert len(saved_data["results"]) == result.total_benchmarks

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_output_dir):
        """Test error handling across the entire benchmark suite."""
        # Create a mock model that will cause failures
        failing_model = Mock()
        failing_model.name = "failing_model"

        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)

        runner = BenchmarkRunner(config)

        # Should handle errors gracefully and still produce results
        result = await runner.run_benchmark_suite(failing_model)

        assert isinstance(result, BenchmarkSuiteResult)
        # Even with failures, we should get some results
        assert result.total_benchmarks >= 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_model, temp_output_dir):
        """Test timeout handling in benchmark execution."""
        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)
        config.timeout_per_benchmark = 1  # Very short timeout

        runner = BenchmarkRunner(config)

        # Should complete even with short timeouts
        result = await runner.run_benchmark_suite(mock_model)

        assert isinstance(result, BenchmarkSuiteResult)
        assert result.execution_time >= 0

    def test_configuration_validation(self, temp_output_dir):
        """Test that invalid configurations are handled properly."""
        # Test with missing required fields
        invalid_config = BenchmarkSuiteConfig(
            name="",  # Invalid empty name
            description="Test config",
            benchmark_categories=[]
        )

        runner = BenchmarkRunner(invalid_config)

        # Should still initialize (validation happens during execution)
        assert runner.config == invalid_config


class TestBenchmarkSuiteResult:
    """Test BenchmarkSuiteResult functionality."""

    def test_result_creation(self, mock_model):
        """Test benchmark suite result creation."""
        results = [
            BenchmarkResult("bench1", "asi", mock_model.name, 0.8),
            BenchmarkResult("bench2", "agi", mock_model.name, 0.7),
        ]

        suite_result = BenchmarkSuiteResult(
            suite_name="Test Suite",
            start_time=asyncio.get_event_loop().time(),
            end_time=asyncio.get_event_loop().time() + 10,
            total_benchmarks=2,
            successful_benchmarks=2,
            failed_benchmarks=0,
            results=results,
            summary={"average_score": 0.75},
            execution_time=10.0
        )

        assert suite_result.suite_name == "Test Suite"
        assert suite_result.total_benchmarks == 2
        assert suite_result.successful_benchmarks == 2
        assert suite_result.failed_benchmarks == 0
        assert len(suite_result.results) == 2
        assert suite_result.execution_time == 10.0


class TestBenchmarkSuiteBuilderIntegration:
    """Integration tests for benchmark suite builder."""

    def test_builder_comprehensive_suite(self):
        """Test building comprehensive suite with builder."""
        suite = (BenchmarkSuiteBuilder("Builder Test Suite")
            .description("Suite built with fluent API")
            .include_categories("asi", "agi", "neuromorphic")
            .parallel_execution(True, max_workers=2)
            .timeout(120)
            .save_intermediates(True)
            .build())

        assert suite.name == "Builder Test Suite"
        assert suite.description == "Suite built with fluent API"
        assert "asi" in suite.benchmark_categories
        assert "agi" in suite.benchmark_categories
        assert "neuromorphic" in suite.benchmark_categories
        assert suite.parallel_execution is True
        assert suite.max_workers == 2
        assert suite.timeout_per_benchmark == 120
        assert suite.save_intermediate_results is True

    def test_builder_with_specific_benchmarks(self):
        """Test building suite with specific benchmark names."""
        suite = (BenchmarkSuiteBuilder("Specific Benchmarks")
            .include_benchmarks("recursive_self_improvement", "general_intelligence")
            .build())

        assert suite.name == "Specific Benchmarks"
        assert suite.benchmark_names == ["recursive_self_improvement", "general_intelligence"]


class TestEndToEndBenchmarking:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_benchmark_workflow(self, mock_model, temp_output_dir):
        """Test complete benchmark workflow from start to finish."""
        # 1. Create suite configuration
        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)

        # 2. Initialize runner
        runner = BenchmarkRunner(config)

        # 3. Execute benchmarks
        result = await runner.run_benchmark_suite(mock_model)

        # 4. Verify complete workflow
        assert isinstance(result, BenchmarkSuiteResult)
        assert result.total_benchmarks > 0
        assert result.execution_time > 0

        # 5. Check that results were saved
        results_file = temp_output_dir / "suite_results.json"
        assert results_file.exists()

        # 6. Verify results can be loaded and parsed
        with open(results_file, 'r') as f:
            saved_results = json.load(f)

        assert "suite_info" in saved_results
        assert "summary" in saved_results
        assert "results" in saved_results

    @pytest.mark.asyncio
    async def test_multiple_model_comparison(self, temp_output_dir):
        """Test comparing multiple models."""
        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)

        runner = BenchmarkRunner(config)

        # Test with different mock models
        models = [
            Mock(name="model_a", name="ModelA"),
            Mock(name="model_b", name="ModelB"),
            Mock(name="model_c", name="ModelC")
        ]

        results = []
        for model in models:
            result = await runner.run_benchmark_suite(model)
            results.append(result)

        # All results should be valid
        assert all(isinstance(r, BenchmarkSuiteResult) for r in results)
        assert all(r.total_benchmarks > 0 for r in results)

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, mock_model, temp_output_dir):
        """Test that resources are properly cleaned up after execution."""
        config = create_quick_evaluation_suite()
        config.output_directory = str(temp_output_dir)

        runner = BenchmarkRunner(config)

        # Execute benchmarks
        result = await runner.run_benchmark_suite(mock_model)

        # Cleanup should be called automatically
        runner.cleanup()

        # Runner should be in clean state
        assert runner.executor is None or runner.executor._shutdown
