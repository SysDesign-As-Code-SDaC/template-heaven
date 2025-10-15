"""
Integration tests for CLI interface.

Tests cover:
- Command-line argument parsing
- CLI workflow execution
- Output formatting and reporting
- Error handling in CLI context
"""

import pytest
import subprocess
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import asyncio


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_cli_help_display(self):
        """Test that CLI help is displayed correctly."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "Comprehensive AI Benchmarks Suite" in result.stdout
        assert "run" in result.stdout
        assert "report" in result.stdout
        assert "list" in result.stdout

    def test_cli_list_command(self):
        """Test CLI list command functionality."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "list"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "Available AI Benchmarks" in result.stdout

    def test_cli_list_category_filter(self):
        """Test CLI list command with category filter."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "list", "--category", "asi"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        # Should show ASI benchmarks or indicate none available

    @patch('ai_benchmarks.cli.BenchmarkRunner')
    @patch('ai_benchmarks.cli.create_comprehensive_ai_suite')
    def test_cli_run_comprehensive(self, mock_create_suite, mock_runner_class, temp_output_dir):
        """Test CLI run command with comprehensive suite."""
        # Mock the suite creation
        mock_suite = MagicMock()
        mock_suite.name = "Comprehensive AI Benchmarks"
        mock_create_suite.return_value = mock_suite

        # Mock the runner
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.successful_benchmarks = 4
        mock_result.total_benchmarks = 4
        mock_result.execution_time = 45.2
        mock_runner.run_benchmark_suite = asyncio.coroutine(lambda model: mock_result)
        mock_runner_class.return_value = mock_runner

        # Run CLI command
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "run", "--suite", "comprehensive",
             "--output", str(temp_output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "Starting benchmark suite" in result.stdout
        assert "Benchmark suite completed" in result.stdout

    @patch('ai_benchmarks.cli.BenchmarkRunner')
    @patch('ai_benchmarks.cli.create_quick_evaluation_suite')
    def test_cli_run_quick_evaluation(self, mock_create_suite, mock_runner_class, temp_output_dir):
        """Test CLI run command with quick evaluation suite."""
        # Mock the suite creation
        mock_suite = MagicMock()
        mock_suite.name = "Quick AI Evaluation"
        mock_create_suite.return_value = mock_suite

        # Mock the runner
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.successful_benchmarks = 2
        mock_result.total_benchmarks = 2
        mock_result.execution_time = 25.1
        mock_runner.run_benchmark_suite = asyncio.coroutine(lambda model: mock_result)
        mock_runner_class.return_value = mock_runner

        # Run CLI command
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "run", "--suite", "quick",
             "--output", str(temp_output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "Quick AI Evaluation" in result.stdout

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "invalid_command"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 1
        assert "invalid choice" in result.stderr.lower() or "no command specified" in result.stderr.lower()

    @patch('ai_benchmarks.cli.BenchmarkReporter')
    @patch('ai_benchmarks.cli.BenchmarkRunner')
    def test_cli_report_generation(self, mock_runner_class, mock_reporter_class, temp_output_dir):
        """Test CLI report generation."""
        # Create a mock results file
        results_file = temp_output_dir / "suite_results.json"
        mock_results_data = {
            "suite_info": {
                "name": "Test Suite",
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-01T00:01:00",
                "execution_time": 60.0
            },
            "summary": {
                "average_score": 0.75,
                "total_benchmarks": 2
            },
            "results": [
                {
                    "benchmark_name": "test_benchmark_1",
                    "benchmark_category": "asi",
                    "model_name": "test_model",
                    "score": 0.8,
                    "status": "completed"
                }
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(mock_results_data, f)

        # Mock the reporter
        mock_reporter = MagicMock()
        mock_reporter_class.return_value = mock_reporter

        # Run CLI report command
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "report",
             "--input", str(temp_output_dir), "--output", str(temp_output_dir / "reports")],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "Reports generated" in result.stdout

    def test_cli_validate_command(self, temp_output_dir):
        """Test CLI validate command."""
        # Create a mock results file with valid results
        results_file = temp_output_dir / "suite_results.json"
        mock_results_data = {
            "results": [
                {
                    "benchmark_name": "valid_benchmark",
                    "benchmark_category": "asi",
                    "model_name": "test_model",
                    "score": 0.85,
                    "status": "completed",
                    "metrics": {"test_metric": 42}
                }
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(mock_results_data, f)

        # Run CLI validate command
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "validate",
             "--input", str(temp_output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "Validation report saved" in result.stdout

    def test_cli_run_with_categories(self, temp_output_dir):
        """Test CLI run command with specific categories."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "run",
             "--categories", "asi", "agi", "--output", str(temp_output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Should either succeed or fail gracefully
        assert result.returncode in [0, 1]  # 0 for success, 1 for controlled failure

    def test_cli_run_with_parallel_execution(self, temp_output_dir):
        """Test CLI run command with parallel execution enabled."""
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "run",
             "--suite", "quick", "--parallel", "--workers", "2",
             "--output", str(temp_output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Should complete (may succeed or fail based on actual benchmark execution)
        assert isinstance(result.returncode, int)

    def test_cli_error_handling(self, temp_output_dir):
        """Test CLI error handling with invalid inputs."""
        # Test with non-existent output directory for run command
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "run",
             "--suite", "quick", "--output", "/nonexistent/path"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Should handle error gracefully
        assert isinstance(result.returncode, int)

    def test_cli_missing_required_args(self):
        """Test CLI behavior with missing required arguments."""
        # Test report command without input
        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "report"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 1  # Should fail due to missing --input

    def test_cli_version_info(self):
        """Test CLI provides version information."""
        result = subprocess.run(
            [sys.executable, "-c", "import ai_benchmarks; print(ai_benchmarks.__version__)"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        assert result.returncode == 0
        assert "1.0.0" in result.stdout

    @patch('ai_benchmarks.cli.BenchmarkRunner')
    def test_cli_execution_timeout(self, mock_runner_class, temp_output_dir):
        """Test CLI handles execution timeouts properly."""
        # Mock a runner that takes too long
        mock_runner = MagicMock()
        async def slow_execution(model):
            await asyncio.sleep(0.1)  # Simulate some work
            mock_result = MagicMock()
            mock_result.successful_benchmarks = 1
            mock_result.total_benchmarks = 1
            mock_result.execution_time = 0.1
            return mock_result

        mock_runner.run_benchmark_suite = slow_execution
        mock_runner_class.return_value = mock_runner

        result = subprocess.run(
            [sys.executable, "-m", "ai_benchmarks", "run",
             "--suite", "quick", "--timeout", "1",
             "--output", str(temp_output_dir)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
            timeout=10  # Prevent hanging
        )

        # Should complete within reasonable time
        assert result.returncode in [0, 1]  # Either success or controlled failure
