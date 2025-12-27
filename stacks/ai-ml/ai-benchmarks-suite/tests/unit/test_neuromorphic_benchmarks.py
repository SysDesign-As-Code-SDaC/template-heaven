"""
Unit tests for neuromorphic computing benchmarks.

Tests cover:
- Spiking neural network benchmark
- Energy monitoring and efficiency
- Temporal processing evaluation
- Neural plasticity assessment
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from ai_benchmarks.core.base_benchmark import BenchmarkConfig, BenchmarkResult
from ai_benchmarks.benchmarks.neuromorphic.spiking_neural_networks import (
    SpikingNeuralNetworkBenchmark,
    EnergyMonitor,
    TemporalProcessor,
    SpikingNetworkMetrics,
    SpikeTrain
)


@pytest.fixture
def neuromorphic_config():
    """Configuration for neuromorphic benchmarks."""
    return BenchmarkConfig(
        name="test_neuromorphic_benchmark",
        category="neuromorphic",
        description="Test neuromorphic benchmark",
        parameters={
            "temporal_sequences": 50,
            "noise_levels": [0.0, 0.1, 0.2],
            "energy_budget": 500,
            "adaptation_iterations": 25
        }
    )


@pytest.fixture
def mock_model():
    """Mock AI model for testing."""
    model = Mock()
    model.name = "test_neuromorphic_model"
    return model


class TestSpikingNeuralNetworkBenchmark:
    """Test spiking neural network benchmark."""

    @pytest.fixture
    def benchmark(self, neuromorphic_config):
        """Create spiking neural network benchmark."""
        return SpikingNeuralNetworkBenchmark(neuromorphic_config)

    def test_initialization(self, benchmark, neuromorphic_config):
        """Test benchmark initialization."""
        assert benchmark.config == neuromorphic_config
        assert benchmark.temporal_sequences == 50
        assert benchmark.energy_budget == 500
        assert benchmark.noise_levels == [0.0, 0.1, 0.2]

    def test_setup(self, benchmark):
        """Test benchmark setup."""
        benchmark.setup()

        assert hasattr(benchmark, 'temporal_datasets')
        assert hasattr(benchmark, 'energy_monitor')
        assert hasattr(benchmark, 'temporal_processor')

        assert isinstance(benchmark.temporal_datasets, list)
        assert len(benchmark.temporal_datasets) > 0

    def test_generate_temporal_datasets(self, benchmark):
        """Test temporal dataset generation."""
        datasets = benchmark._generate_temporal_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) > 0

        for dataset in datasets:
            assert "pattern_type" in dataset
            assert "spike_train" in dataset
            assert "duration" in dataset
            assert "expected_response" in dataset

            assert isinstance(dataset["spike_train"], SpikeTrain)

    def test_generate_spike_pattern_regular(self, benchmark):
        """Test regular spiking pattern generation."""
        pattern = benchmark._generate_spike_pattern("regular_spiking")

        assert isinstance(pattern, SpikeTrain)
        assert len(pattern.timestamps) > 0
        assert len(pattern.neuron_ids) == len(pattern.timestamps)
        assert len(pattern.intensities) == len(pattern.timestamps)

    def test_generate_spike_pattern_bursty(self, benchmark):
        """Test bursty spiking pattern generation."""
        pattern = benchmark._generate_spike_pattern("bursty_spiking")

        assert isinstance(pattern, SpikeTrain)
        assert len(pattern.timestamps) > 0

    def test_generate_spike_pattern_irregular(self, benchmark):
        """Test irregular spiking pattern generation."""
        pattern = benchmark._generate_spike_pattern("irregular_spiking")

        assert isinstance(pattern, SpikeTrain)
        assert len(pattern.timestamps) > 0

    def test_evaluate_temporal_processing(self, benchmark, mock_model):
        """Test temporal processing evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_temporal_processing(mock_model)

        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "correct_predictions" in results
        assert "total_predictions" in results

        assert isinstance(results["accuracy"], float)
        assert 0 <= results["accuracy"] <= 1

    def test_evaluate_energy_efficiency(self, benchmark, mock_model):
        """Test energy efficiency evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_energy_efficiency(mock_model)

        assert isinstance(results, dict)
        assert "total_energy" in results
        assert "average_energy" in results
        assert "efficiency" in results
        assert "energy_budget" in results

        assert results["energy_budget"] == benchmark.energy_budget

    def test_evaluate_event_processing(self, benchmark, mock_model):
        """Test event processing evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_event_processing(mock_model)

        assert isinstance(results, dict)
        assert "spike_efficiency" in results
        assert "total_spikes_processed" in results
        assert "useful_spikes" in results

    def test_evaluate_synaptic_plasticity(self, benchmark, mock_model):
        """Test synaptic plasticity evaluation."""
        results = benchmark._evaluate_synaptic_plasticity(mock_model)

        assert isinstance(results, dict)
        assert "adaptation_rate" in results
        assert "plasticity_measure" in results
        assert "weight_changes" in results

    def test_evaluate_noise_robustness(self, benchmark, mock_model):
        """Test noise robustness evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_noise_robustness(mock_model)

        assert isinstance(results, dict)
        assert "robustness_scores" in results
        assert "noise_levels" in results
        assert "average_robustness" in results

        assert len(results["robustness_scores"]) == len(benchmark.noise_levels)

    def test_evaluate_realtime_performance(self, benchmark, mock_model):
        """Test real-time performance evaluation."""
        benchmark.setup()
        results = benchmark._evaluate_realtime_performance(mock_model)

        assert isinstance(results, dict)
        assert "processing_times" in results
        assert "average_time" in results
        assert "processing_speed" in results

    def test_calculate_spiking_metrics(self, benchmark):
        """Test spiking metrics calculation."""
        temporal_results = {"accuracy": 0.85}
        energy_results = {"efficiency": 0.75}
        event_results = {"spike_efficiency": 0.8}
        plasticity_results = {"adaptation_rate": 0.6, "plasticity_measure": 0.3}
        robustness_results = {"average_robustness": 0.7}
        realtime_results = {"processing_speed": 0.9}

        metrics = benchmark._calculate_spiking_metrics(
            temporal_results, energy_results, event_results,
            plasticity_results, robustness_results, realtime_results
        )

        assert isinstance(metrics, SpikingNetworkMetrics)
        assert metrics.temporal_accuracy == 0.85
        assert metrics.energy_efficiency == 0.75
        assert metrics.spike_efficiency == 0.8
        assert metrics.noise_robustness == 0.7
        assert metrics.real_time_performance == 0.9

    def test_calculate_overall_score(self, benchmark):
        """Test overall score calculation."""
        metrics = SpikingNetworkMetrics(
            temporal_accuracy=0.8,
            energy_efficiency=0.7,
            spike_efficiency=0.75,
            adaptation_rate=0.6,
            noise_robustness=0.8,
            real_time_performance=0.85,
            synaptic_plasticity=0.4
        )

        score = benchmark._calculate_overall_score(metrics)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_validate_result(self, benchmark):
        """Test result validation."""
        valid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="neuromorphic",
            model_name="model",
            score=0.8,
            metrics={
                "temporal_accuracy": 0.8,
                "energy_efficiency": 0.7,
                "spike_efficiency": 0.75,
                "adaptation_rate": 0.6,
                "noise_robustness": 0.8,
                "realtime_performance": 0.85
            }
        )

        assert benchmark.validate_result(valid_result)

        # Test invalid result
        invalid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="neuromorphic",
            model_name="model",
            score=0.8,
            metrics={}  # Missing required metrics
        )

        assert not benchmark.validate_result(invalid_result)

    def test_cleanup(self, benchmark):
        """Test benchmark cleanup."""
        benchmark.setup()
        benchmark.cleanup()

        assert len(benchmark.temporal_datasets) == 0


class TestSpikeTrain:
    """Test SpikeTrain dataclass."""

    def test_spike_train_creation(self):
        """Test spike train creation."""
        timestamps = np.array([1.0, 2.5, 3.8])
        neuron_ids = np.array([0, 1, 0])
        intensities = np.array([1.0, 0.8, 1.0])

        spike_train = SpikeTrain(
            timestamps=timestamps,
            neuron_ids=neuron_ids,
            intensities=intensities
        )

        assert np.array_equal(spike_train.timestamps, timestamps)
        assert np.array_equal(spike_train.neuron_ids, neuron_ids)
        assert np.array_equal(spike_train.intensities, intensities)

    def test_spike_train_without_intensities(self):
        """Test spike train creation without intensities."""
        timestamps = np.array([1.0, 2.5, 3.8])
        neuron_ids = np.array([0, 1, 0])

        spike_train = SpikeTrain(
            timestamps=timestamps,
            neuron_ids=neuron_ids
        )

        assert spike_train.intensities is None
        assert len(spike_train.timestamps) == 3
        assert len(spike_train.neuron_ids) == 3


class TestEnergyMonitor:
    """Test energy monitoring functionality."""

    @pytest.fixture
    def energy_monitor(self):
        """Create energy monitor instance."""
        return EnergyMonitor()

    def test_initialization(self, energy_monitor):
        """Test energy monitor initialization."""
        assert energy_monitor.energy_units == 0

    def test_measure_energy(self, energy_monitor):
        """Test energy measurement."""
        def dummy_computation():
            return sum(range(100))

        energy = energy_monitor.measure_energy(dummy_computation)

        assert isinstance(energy, (int, float))
        assert energy > 0

    def test_reset(self, energy_monitor):
        """Test energy monitor reset."""
        energy_monitor.energy_units = 100
        energy_monitor.reset()

        assert energy_monitor.energy_units == 0


class TestTemporalProcessor:
    """Test temporal processing functionality."""

    @pytest.fixture
    def temporal_processor(self):
        """Create temporal processor instance."""
        return TemporalProcessor()

    def test_initialization(self, temporal_processor):
        """Test temporal processor initialization."""
        assert temporal_processor.buffer_size == 1000

    def test_process_temporal_pattern(self, temporal_processor):
        """Test temporal pattern processing."""
        spike_train = SpikeTrain(
            timestamps=np.array([1.0, 2.0, 3.0]),
            neuron_ids=np.array([0, 1, 0])
        )

        result = temporal_processor.process_temporal_pattern(spike_train)

        assert isinstance(result, dict)
        assert "pattern_detected" in result
        assert "correlation_strength" in result
        assert "temporal_complexity" in result


class TestSpikingNetworkMetrics:
    """Test SpikingNetworkMetrics dataclass."""

    def test_metrics_creation(self):
        """Test spiking network metrics creation."""
        metrics = SpikingNetworkMetrics(
            temporal_accuracy=0.85,
            energy_efficiency=0.75,
            spike_efficiency=0.8,
            adaptation_rate=0.7,
            noise_robustness=0.9,
            real_time_performance=0.8,
            synaptic_plasticity=0.6
        )

        assert metrics.temporal_accuracy == 0.85
        assert metrics.energy_efficiency == 0.75
        assert metrics.spike_efficiency == 0.8
        assert metrics.adaptation_rate == 0.7
        assert metrics.noise_robustness == 0.9
        assert metrics.real_time_performance == 0.8
        assert metrics.synaptic_plasticity == 0.6
