"""
Unit tests for AGI (Artificial General Intelligence) benchmarks.

Tests cover:
- General intelligence benchmark
- Cognitive domain evaluation
- Transfer learning assessment
- Knowledge integration
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from ai_benchmarks.core.base_benchmark import BenchmarkConfig, BenchmarkResult
from ai_benchmarks.benchmarks.agi.general_intelligence import (
    GeneralIntelligenceBenchmark,
    CognitiveDomain,
    AGIMetrics
)


@pytest.fixture
def agi_config():
    """Configuration for AGI benchmarks."""
    return BenchmarkConfig(
        name="test_agi_benchmark",
        category="agi",
        description="Test AGI benchmark",
        parameters={
            "domains_count": 4,
            "tasks_per_domain": 3,
            "adaptation_rounds": 2
        }
    )


@pytest.fixture
def mock_model():
    """Mock AI model for testing."""
    model = Mock()
    model.name = "test_agi_model"
    return model


class TestGeneralIntelligenceBenchmark:
    """Test general intelligence benchmark."""

    @pytest.fixture
    def benchmark(self, agi_config):
        """Create general intelligence benchmark."""
        return GeneralIntelligenceBenchmark(agi_config)

    def test_initialization(self, benchmark, agi_config):
        """Test benchmark initialization."""
        assert benchmark.config == agi_config
        assert benchmark.domains_count == 4
        assert benchmark.tasks_per_domain == 3
        assert benchmark.adaptation_rounds == 2

    def test_setup(self, benchmark):
        """Test benchmark setup."""
        benchmark.setup()

        assert hasattr(benchmark, 'cognitive_domains')
        assert hasattr(benchmark, 'knowledge_base')
        assert hasattr(benchmark, 'domain_performances')
        assert hasattr(benchmark, 'transfer_evaluations')

        assert isinstance(benchmark.cognitive_domains, list)
        assert len(benchmark.cognitive_domains) > 0

    def test_initialize_cognitive_domains(self, benchmark):
        """Test cognitive domain initialization."""
        domains = benchmark._initialize_cognitive_domains()

        assert isinstance(domains, list)
        assert len(domains) == 8  # Default number of domains

        for domain in domains:
            assert isinstance(domain, CognitiveDomain)
            assert hasattr(domain, 'name')
            assert hasattr(domain, 'description')
            assert hasattr(domain, 'tasks')
            assert hasattr(domain, 'difficulty_levels')
            assert hasattr(domain, 'evaluation_metrics')

    def test_evaluate_multi_domain_learning(self, benchmark, mock_model):
        """Test multi-domain learning evaluation."""
        benchmark.setup()
        domain_results = benchmark._evaluate_multi_domain_learning(mock_model)

        assert isinstance(domain_results, dict)
        assert len(domain_results) > 0

        for domain_name, task_scores in domain_results.items():
            assert isinstance(task_scores, list)
            assert len(task_scores) == benchmark.tasks_per_domain
            assert all(isinstance(score, float) for score in task_scores)
            assert all(0 <= score <= 1 for score in task_scores)

    def test_evaluate_transfer_learning(self, benchmark, mock_model):
        """Test transfer learning evaluation."""
        benchmark.setup()
        transfer_results = benchmark._evaluate_transfer_learning(mock_model)

        assert isinstance(transfer_results, dict)
        assert "transfer_tests" in transfer_results
        assert "overall_score" in transfer_results

        assert isinstance(transfer_results["transfer_tests"], list)
        assert isinstance(transfer_results["overall_score"], float)
        assert 0 <= transfer_results["overall_score"] <= 1

    def test_evaluate_cognitive_flexibility(self, benchmark, mock_model):
        """Test cognitive flexibility evaluation."""
        flexibility_score = benchmark._evaluate_cognitive_flexibility(mock_model)

        assert isinstance(flexibility_score, float)
        assert 0 <= flexibility_score <= 1

    def test_evaluate_adaptation_capability(self, benchmark, mock_model):
        """Test adaptation capability evaluation."""
        adaptation_results = benchmark._evaluate_adaptation_capability(mock_model)

        assert isinstance(adaptation_results, dict)
        assert "rounds" in adaptation_results
        assert "average_adaptation_rate" in adaptation_results
        assert "learning_acceleration" in adaptation_results

        assert isinstance(adaptation_results["rounds"], list)
        assert len(adaptation_results["rounds"]) == benchmark.adaptation_rounds

    def test_evaluate_knowledge_integration(self, benchmark, mock_model):
        """Test knowledge integration evaluation."""
        integration_score = benchmark._evaluate_knowledge_integration(mock_model)

        assert isinstance(integration_score, float)
        assert 0 <= integration_score <= 1

    def test_calculate_agi_metrics(self, benchmark):
        """Test AGI metrics calculation."""
        # Mock results for testing
        domain_results = {
            "mathematical_reasoning": [0.8, 0.7, 0.9],
            "linguistic_understanding": [0.6, 0.8, 0.7],
            "spatial_reasoning": [0.9, 0.8, 0.9],
            "scientific_reasoning": [0.7, 0.6, 0.8]
        }

        transfer_results = {"overall_score": 0.75}
        flexibility_score = 0.8
        adaptation_results = {
            "average_adaptation_rate": 0.7,
            "rounds": [{"adaptation_rate": 0.7}, {"adaptation_rate": 0.7}]
        }
        integration_score = 0.85

        metrics = benchmark._calculate_agi_metrics(
            domain_results, transfer_results, flexibility_score,
            adaptation_results, integration_score
        )

        assert isinstance(metrics, AGIMetrics)
        assert hasattr(metrics, 'cognitive_flexibility')
        assert hasattr(metrics, 'learning_efficiency')
        assert hasattr(metrics, 'transfer_learning_score')
        assert hasattr(metrics, 'abstract_reasoning')

        # Check that all metrics are in valid range
        for attr_name in dir(metrics):
            if not attr_name.startswith('_'):
                value = getattr(metrics, attr_name)
                if isinstance(value, (int, float)):
                    assert 0 <= value <= 1, f"Metric {attr_name} out of range: {value}"

    def test_calculate_overall_agi_score(self, benchmark):
        """Test overall AGI score calculation."""
        metrics = AGIMetrics(
            cognitive_flexibility=0.8,
            learning_efficiency=0.75,
            transfer_learning_score=0.7,
            commonsense_reasoning=0.8,
            abstract_reasoning=0.85,
            multi_modal_understanding=0.7,
            adaptation_rate=0.75,
            knowledge_integration=0.8
        )

        score = benchmark._calculate_overall_agi_score(metrics)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_generate_domain_tasks(self, benchmark):
        """Test domain task generation."""
        domain = CognitiveDomain(
            name="test_domain",
            description="Test cognitive domain",
            tasks=["task1", "task2"],
            difficulty_levels=["easy", "medium"],
            evaluation_metrics=["accuracy", "speed"]
        )

        tasks = benchmark._generate_domain_tasks(domain)

        assert isinstance(tasks, list)
        assert len(tasks) == benchmark.tasks_per_domain

        for task in tasks:
            assert "domain" in task
            assert "type" in task
            assert "difficulty" in task
            assert "description" in task
            assert task["domain"] == domain.name
            assert task["type"] in domain.tasks
            assert task["difficulty"] in domain.difficulty_levels

    def test_evaluate_task_performance(self, benchmark, mock_model):
        """Test task performance evaluation."""
        task = {
            "domain": "test_domain",
            "type": "test_task",
            "difficulty": "medium",
            "description": "Test task"
        }

        score = benchmark._evaluate_task_performance(mock_model, task, benchmark.cognitive_domains[0])

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_test_domain_transfer(self, benchmark, mock_model):
        """Test domain transfer evaluation."""
        transfer_score = benchmark._test_domain_transfer(mock_model, "domain1", "domain2")

        assert isinstance(transfer_score, float)
        assert 0 <= transfer_score <= 1

    def test_test_cognitive_switching(self, benchmark, mock_model):
        """Test cognitive switching evaluation."""
        scenario = "analytical_to_creative"
        score = benchmark._test_cognitive_switching(mock_model, scenario)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_generate_adaptation_challenge(self, benchmark):
        """Test adaptation challenge generation."""
        round_num = 1
        challenge = benchmark._generate_adaptation_challenge(round_num)

        assert isinstance(challenge, dict)
        assert "round" in challenge
        assert "complexity" in challenge
        assert challenge["round"] == round_num

    def test_measure_adaptation(self, benchmark, mock_model):
        """Test adaptation measurement."""
        challenge = {"round": 1, "complexity": 2}
        metrics = benchmark._measure_adaptation(mock_model, challenge)

        assert isinstance(metrics, dict)
        assert "adaptation_rate" in metrics
        assert "performance_gain" in metrics
        assert "stability" in metrics

        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_calculate_learning_acceleration(self, benchmark):
        """Test learning acceleration calculation."""
        adaptation_results = [
            {"adaptation_rate": 0.6},
            {"adaptation_rate": 0.7},
            {"adaptation_rate": 0.8}
        ]

        acceleration = benchmark._calculate_learning_acceleration(adaptation_results)

        assert isinstance(acceleration, (int, float))

    def test_test_knowledge_synthesis(self, benchmark, mock_model):
        """Test knowledge synthesis evaluation."""
        scenario = "scientific_method_meets_ethics"
        score = benchmark._test_knowledge_synthesis(mock_model, scenario)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_validate_result(self, benchmark):
        """Test result validation."""
        valid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="agi",
            model_name="model",
            score=0.8,
            metrics={
                "domain_results": {"domain1": [0.8, 0.7]},
                "transfer_learning_score": 0.75,
                "cognitive_flexibility": 0.8,
                "adaptation_results": {"rounds": []},
                "knowledge_integration": 0.85
            }
        )

        assert benchmark.validate_result(valid_result)

        # Test invalid result
        invalid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="agi",
            model_name="model",
            score=0.8,
            metrics={}  # Missing required metrics
        )

        assert not benchmark.validate_result(invalid_result)

    def test_cleanup(self, benchmark):
        """Test benchmark cleanup."""
        benchmark.setup()
        benchmark.cleanup()

        assert len(benchmark.domain_performances) == 0
        assert len(benchmark.transfer_evaluations) == 0
        assert len(benchmark.knowledge_base) == 0


class TestCognitiveDomain:
    """Test CognitiveDomain dataclass."""

    def test_domain_creation(self):
        """Test cognitive domain creation."""
        domain = CognitiveDomain(
            name="test_domain",
            description="Test cognitive domain",
            tasks=["task1", "task2", "task3"],
            difficulty_levels=["easy", "medium", "hard"],
            evaluation_metrics=["accuracy", "speed", "robustness"]
        )

        assert domain.name == "test_domain"
        assert domain.description == "Test cognitive domain"
        assert len(domain.tasks) == 3
        assert len(domain.difficulty_levels) == 3
        assert len(domain.evaluation_metrics) == 3

    def test_domain_attributes(self):
        """Test domain attribute access."""
        domain = CognitiveDomain(
            name="mathematical_reasoning",
            description="Numerical and symbolic reasoning",
            tasks=["arithmetic", "algebra"],
            difficulty_levels=["basic", "advanced"],
            evaluation_metrics=["correctness", "efficiency"]
        )

        assert domain.name == "mathematical_reasoning"
        assert "arithmetic" in domain.tasks
        assert "basic" in domain.difficulty_levels
        assert "correctness" in domain.evaluation_metrics


class TestAGIMetrics:
    """Test AGI metrics dataclass."""

    def test_metrics_creation(self):
        """Test AGI metrics creation."""
        metrics = AGIMetrics(
            cognitive_flexibility=0.8,
            learning_efficiency=0.75,
            transfer_learning_score=0.7,
            commonsense_reasoning=0.85,
            abstract_reasoning=0.8,
            multi_modal_understanding=0.7,
            adaptation_rate=0.75,
            knowledge_integration=0.9
        )

        assert metrics.cognitive_flexibility == 0.8
        assert metrics.learning_efficiency == 0.75
        assert metrics.transfer_learning_score == 0.7
        assert metrics.commonsense_reasoning == 0.85
        assert metrics.abstract_reasoning == 0.8
        assert metrics.multi_modal_understanding == 0.7
        assert metrics.adaptation_rate == 0.75
        assert metrics.knowledge_integration == 0.9

    def test_metrics_range_validation(self):
        """Test that metrics are in valid range."""
        metrics = AGIMetrics(
            cognitive_flexibility=1.5,  # Invalid
            learning_efficiency=0.75,
            transfer_learning_score=0.7,
            commonsense_reasoning=0.85,
            abstract_reasoning=0.8,
            multi_modal_understanding=0.7,
            adaptation_rate=0.75,
            knowledge_integration=0.9
        )

        # Note: This doesn't enforce range validation automatically
        # Range validation should be done in the benchmark logic
        assert metrics.cognitive_flexibility == 1.5
