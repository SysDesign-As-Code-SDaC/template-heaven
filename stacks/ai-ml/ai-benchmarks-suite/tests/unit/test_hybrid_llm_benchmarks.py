"""
Unit tests for hybrid LLM benchmarks.

Tests cover:
- Hybrid architecture benchmark
- Ensemble coordination
- Multi-modal integration
- Failure simulation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from ai_benchmarks.core.base_benchmark import BenchmarkConfig, BenchmarkResult
from ai_benchmarks.benchmarks.hybrid_llms.hybrid_architecture import (
    HybridArchitectureBenchmark,
    EnsembleCoordinator,
    FailureSimulator,
    HybridArchitectureMetrics,
    ArchitectureComponent
)


@pytest.fixture
def hybrid_config():
    """Configuration for hybrid LLM benchmarks."""
    return BenchmarkConfig(
        name="test_hybrid_benchmark",
        category="hybrid_llms",
        description="Test hybrid LLM benchmark",
        parameters={
            "ensemble_size": 2,
            "modalities": ["text", "vision", "structured"],
            "failure_scenarios": 3
        }
    )


@pytest.fixture
def mock_model():
    """Mock AI model for testing."""
    model = Mock()
    model.name = "test_hybrid_model"
    return model


class TestHybridArchitectureBenchmark:
    """Test hybrid architecture benchmark."""

    @pytest.fixture
    def benchmark(self, hybrid_config):
        """Create hybrid architecture benchmark."""
        return HybridArchitectureBenchmark(hybrid_config)

    def test_initialization(self, benchmark, hybrid_config):
        """Test benchmark initialization."""
        assert benchmark.config == hybrid_config
        assert benchmark.ensemble_size == 2
        assert benchmark.modalities == ["text", "vision", "structured"]
        assert benchmark.failure_scenarios == 3

    def test_setup(self, benchmark):
        """Test benchmark setup."""
        benchmark.setup()

        assert hasattr(benchmark, 'architecture_components')
        assert hasattr(benchmark, 'multi_modal_datasets')
        assert hasattr(benchmark, 'ensemble_coordinator')
        assert hasattr(benchmark, 'failure_simulator')

        assert isinstance(benchmark.architecture_components, list)
        assert len(benchmark.architecture_components) > 0
        assert isinstance(benchmark.multi_modal_datasets, list)

    def test_initialize_components(self, benchmark):
        """Test architecture component initialization."""
        components = benchmark._initialize_components()

        assert isinstance(components, list)
        assert len(components) > 0

        for component in components:
            assert isinstance(component, ArchitectureComponent)
            assert hasattr(component, 'name')
            assert hasattr(component, 'type')
            assert hasattr(component, 'modality')
            assert hasattr(component, 'parameters')
            assert hasattr(component, 'performance_profile')

    def test_generate_multi_modal_datasets(self, benchmark):
        """Test multi-modal dataset generation."""
        datasets = benchmark._generate_multi_modal_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) > 0

        for dataset in datasets:
            assert "type" in dataset
            assert "task" in dataset
            if "inputs" in dataset:
                assert isinstance(dataset["inputs"], dict)

    def test_evaluate_ensemble_performance(self, benchmark, mock_model):
        """Test ensemble performance evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_ensemble_performance(mock_model)

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "diversity_score" in results
        assert "improvement_over_best_single" in results

        assert isinstance(results["accuracy"], float)
        assert 0 <= results["accuracy"] <= 1

    def test_evaluate_multi_modal_integration(self, benchmark, mock_model):
        """Test multi-modal integration evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_multi_modal_integration(mock_model)

        assert isinstance(results, dict)
        assert "efficiency" in results
        assert "fusion_efficiency" in results
        assert "modality_coverage" in results

    def test_evaluate_adaptive_switching(self, benchmark, mock_model):
        """Test adaptive switching evaluation."""
        results = benchmark._evaluate_adaptive_switching(mock_model)

        assert isinstance(results, dict)
        assert "switching_accuracy" in results
        assert "scenarios_tested" in results
        assert "adaptive_efficiency" in results

        assert 0 <= results["switching_accuracy"] <= 1

    def test_evaluate_cross_modal_transfer(self, benchmark, mock_model):
        """Test cross-modal transfer evaluation."""
        results = benchmark._evaluate_cross_modal_transfer(mock_model)

        assert isinstance(results, dict)
        assert "transfer_score" in results
        assert "transfer_pairs_tested" in results
        assert "transfer_consistency" in results

    def test_evaluate_hybrid_reasoning(self, benchmark, mock_model):
        """Test hybrid reasoning evaluation."""
        results = benchmark._evaluate_hybrid_reasoning(mock_model)

        assert isinstance(results, dict)
        assert "coherence_score" in results
        assert "reasoning_tasks" in results
        assert "reasoning_depth" in results

    def test_evaluate_robustness(self, benchmark, mock_model):
        """Test robustness evaluation."""
        results = benchmark._evaluate_robustness(mock_model)

        assert isinstance(results, dict)
        assert "robustness_score" in results
        assert "failure_scenarios" in results
        assert "graceful_degradation" in results

    def test_evaluate_computational_efficiency(self, benchmark, mock_model):
        """Test computational efficiency evaluation."""
        results = benchmark._evaluate_computational_efficiency(mock_model)

        assert isinstance(results, dict)
        assert "efficiency_ratio" in results
        assert "efficiency_improvement" in results
        assert "resource_breakdown" in results

    def test_calculate_hybrid_metrics(self, benchmark):
        """Test hybrid metrics calculation."""
        ensemble_results = {"accuracy": 0.85}
        integration_results = {"efficiency": 0.8}
        switching_results = {"switching_accuracy": 0.75}
        transfer_results = {"transfer_score": 0.7}
        reasoning_results = {"coherence_score": 0.8}
        robustness_results = {"robustness_score": 0.9}
        efficiency_results = {"efficiency_ratio": 0.85}

        metrics = benchmark._calculate_hybrid_metrics(
            ensemble_results, integration_results, switching_results,
            transfer_results, reasoning_results, robustness_results, efficiency_results
        )

        assert isinstance(metrics, HybridArchitectureMetrics)
        assert metrics.ensemble_accuracy == 0.85
        assert metrics.integration_efficiency == 0.8
        assert metrics.adaptive_switching == 0.75
        assert metrics.cross_modal_transfer == 0.7
        assert metrics.reasoning_coherence == 0.8
        assert metrics.robustness_to_failure == 0.9
        assert metrics.computational_efficiency == 0.85

    def test_calculate_overall_score(self, benchmark):
        """Test overall score calculation."""
        metrics = HybridArchitectureMetrics(
            ensemble_accuracy=0.85,
            integration_efficiency=0.8,
            cross_modal_transfer=0.7,
            adaptive_switching=0.75,
            reasoning_coherence=0.8,
            computational_efficiency=0.85,
            robustness_to_failure=0.9
        )

        score = benchmark._calculate_overall_score(metrics)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_validate_result(self, benchmark):
        """Test result validation."""
        valid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="hybrid_llms",
            model_name="model",
            score=0.8,
            metrics={
                "ensemble_accuracy": 0.85,
                "integration_efficiency": 0.8,
                "adaptive_switching_score": 0.75,
                "cross_modal_transfer": 0.7,
                "reasoning_coherence": 0.8,
                "failure_robustness": 0.9,
                "computational_efficiency": 0.85
            }
        )

        assert benchmark.validate_result(valid_result)

        # Test invalid result
        invalid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="hybrid_llms",
            model_name="model",
            score=0.8,
            metrics={}  # Missing required metrics
        )

        assert not benchmark.validate_result(invalid_result)

    def test_cleanup(self, benchmark):
        """Test benchmark cleanup."""
        benchmark.setup()
        benchmark.cleanup()

        assert len(benchmark.architecture_components) == 0
        assert len(benchmark.multi_modal_datasets) == 0


class TestEnsembleCoordinator:
    """Test ensemble coordination functionality."""

    @pytest.fixture
    def ensemble_coordinator(self):
        """Create ensemble coordinator instance."""
        return EnsembleCoordinator(ensemble_size=3)

    def test_initialization(self, ensemble_coordinator):
        """Test ensemble coordinator initialization."""
        assert ensemble_coordinator.ensemble_size == 3

    def test_predict_ensemble(self, ensemble_coordinator, mock_model):
        """Test ensemble prediction."""
        dataset = {"type": "test", "task": "classification"}

        result = ensemble_coordinator.predict_ensemble(mock_model, dataset)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert "diversity_score" in result

    def test_reset(self, ensemble_coordinator):
        """Test ensemble coordinator reset."""
        # Should not raise any exceptions
        ensemble_coordinator.reset()


class TestFailureSimulator:
    """Test failure simulation functionality."""

    @pytest.fixture
    def failure_simulator(self):
        """Create failure simulator instance."""
        return FailureSimulator()

    def test_simulate_failure(self, failure_simulator, mock_model):
        """Test failure simulation."""
        components = [
            ArchitectureComponent(
                name="comp1",
                type="transformer",
                modality="text",
                parameters={},
                performance_profile={"accuracy": 0.8}
            ),
            ArchitectureComponent(
                name="comp2",
                type="cnn",
                modality="vision",
                parameters={},
                performance_profile={"accuracy": 0.7}
            )
        ]

        failed_model = failure_simulator.simulate_failure(mock_model, components)

        # The failed model should still be a mock (simulation)
        assert failed_model is not None

    def test_reset(self, failure_simulator):
        """Test failure simulator reset."""
        # Should not raise any exceptions
        failure_simulator.reset()


class TestArchitectureComponent:
    """Test ArchitectureComponent dataclass."""

    def test_component_creation(self):
        """Test architecture component creation."""
        component = ArchitectureComponent(
            name="test_transformer",
            type="transformer",
            modality="text",
            parameters={"layers": 12, "heads": 8},
            performance_profile={
                "accuracy": 0.85,
                "speed": 0.7,
                "memory": 0.6
            }
        )

        assert component.name == "test_transformer"
        assert component.type == "transformer"
        assert component.modality == "text"
        assert component.parameters["layers"] == 12
        assert component.performance_profile["accuracy"] == 0.85

    def test_component_attributes(self):
        """Test component attribute access."""
        component = ArchitectureComponent(
            name="vision_cnn",
            type="cnn",
            modality="vision",
            parameters={"filters": [32, 64, 128]},
            performance_profile={"accuracy": 0.82, "speed": 0.8}
        )

        assert component.name == "vision_cnn"
        assert component.type == "cnn"
        assert component.modality == "vision"
        assert 32 in component.parameters["filters"]
        assert component.performance_profile["speed"] == 0.8


class TestHybridArchitectureMetrics:
    """Test HybridArchitectureMetrics dataclass."""

    def test_metrics_creation(self):
        """Test hybrid architecture metrics creation."""
        metrics = HybridArchitectureMetrics(
            ensemble_accuracy=0.85,
            integration_efficiency=0.8,
            cross_modal_transfer=0.75,
            adaptive_switching=0.7,
            reasoning_coherence=0.8,
            computational_efficiency=0.85,
            robustness_to_failure=0.9
        )

        assert metrics.ensemble_accuracy == 0.85
        assert metrics.integration_efficiency == 0.8
        assert metrics.cross_modal_transfer == 0.75
        assert metrics.adaptive_switching == 0.7
        assert metrics.reasoning_coherence == 0.8
        assert metrics.computational_efficiency == 0.85
        assert metrics.robustness_to_failure == 0.9
