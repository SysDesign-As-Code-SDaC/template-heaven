"""
Unit tests for ASI (Artificial Super Intelligence) benchmarks.

Tests cover:
- Recursive self-improvement benchmark
- Universal problem solver benchmark
- Safety monitoring and validation
- Performance metrics calculation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from ai_benchmarks.core.base_benchmark import BenchmarkConfig, BenchmarkResult
from ai_benchmarks.benchmarks.asi.recursive_self_improvement import (
    RecursiveSelfImprovementBenchmark,
    SafetyMonitor,
    SelfImprovementMetrics
)
from ai_benchmarks.benchmarks.asi.universal_problem_solver import (
    UniversalProblemSolverBenchmark,
    MathematicalEvaluator,
    LogicalEvaluator,
    ScientificEvaluator,
    StrategicEvaluator,
    CreativeEvaluator
)


@pytest.fixture
def asi_config():
    """Configuration for ASI benchmarks."""
    return BenchmarkConfig(
        name="test_asi_benchmark",
        category="asi",
        description="Test ASI benchmark",
        parameters={
            "iterations": 3,
            "safety_threshold": 0.9,
            "max_depth": 3,
            "max_problems_per_domain": 2,
            "time_limit": 30
        }
    )


@pytest.fixture
def mock_model():
    """Mock AI model for testing."""
    model = Mock()
    model.name = "test_asi_model"
    return model


class TestRecursiveSelfImprovementBenchmark:
    """Test recursive self-improvement benchmark."""

    @pytest.fixture
    def benchmark(self, asi_config):
        """Create recursive self-improvement benchmark."""
        return RecursiveSelfImprovementBenchmark(asi_config)

    def test_initialization(self, benchmark, asi_config):
        """Test benchmark initialization."""
        assert benchmark.config == asi_config
        assert benchmark.improvement_iterations == 3
        assert benchmark.safety_threshold == 0.9
        assert benchmark.max_recursive_depth == 3

    def test_setup(self, benchmark):
        """Test benchmark setup."""
        benchmark.setup()

        assert hasattr(benchmark, 'test_problems')
        assert hasattr(benchmark, 'performance_history')
        assert hasattr(benchmark, 'safety_monitor')
        assert isinstance(benchmark.test_problems, list)
        assert len(benchmark.test_problems) > 0

    def test_generate_test_problems(self, benchmark):
        """Test test problem generation."""
        problems = benchmark._generate_test_problems()

        assert isinstance(problems, list)
        assert len(problems) > 0

        for problem in problems:
            assert "type" in problem
            assert "complexity" in problem
            assert problem["type"] in ["optimization", "classification", "reinforcement_learning"]

    def test_evaluate_performance(self, benchmark, mock_model):
        """Test performance evaluation."""
        benchmark.setup()
        score = benchmark._evaluate_performance(mock_model)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_analyze_performance(self, benchmark, mock_model):
        """Test performance analysis."""
        benchmark.setup()
        analysis = benchmark._analyze_performance(mock_model)

        required_keys = ["weaknesses", "strengths", "improvement_potential", "architectural_insights"]
        for key in required_keys:
            assert key in analysis

    def test_generate_improvements(self, benchmark):
        """Test improvement generation."""
        analysis = {
            "weaknesses": [
                {"component": "optimizer", "impact": 0.3},
                {"component": "architecture", "impact": 0.2}
            ],
            "strengths": [],
            "improvement_potential": 0.8,
            "architectural_insights": {}
        }

        improvements = benchmark._generate_improvements(analysis)

        assert isinstance(improvements, list)
        assert len(improvements) == 2  # One for each weakness

        for improvement in improvements:
            required_keys = ["type", "target", "description", "estimated_impact", "risk_level"]
            for key in required_keys:
                assert key in improvement

    def test_calculate_improvement_metrics(self, benchmark):
        """Test improvement metrics calculation."""
        initial_perf = 0.6
        final_perf = 0.85

        # Simulate some performance history
        benchmark.performance_history = [0.6, 0.7, 0.8, 0.85]

        metrics = benchmark._calculate_improvement_metrics(initial_perf, final_perf)

        assert isinstance(metrics, SelfImprovementMetrics)
        assert metrics.initial_performance == 0.6
        assert metrics.final_performance == 0.85
        assert metrics.improvement_rate > 0
        assert metrics.stability_score >= 0
        assert metrics.recursive_depth == 3

    def test_calculate_overall_score(self, benchmark):
        """Test overall score calculation."""
        metrics = SelfImprovementMetrics(
            initial_performance=0.5,
            final_performance=0.8,
            improvement_rate=0.6,
            convergence_speed=0.1,
            stability_score=0.9,
            safety_violations=0,
            meta_learning_efficiency=0.7,
            recursive_depth=3
        )

        score = benchmark._calculate_overall_score(metrics, 0, 5)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_validate_result(self, benchmark):
        """Test result validation."""
        valid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="asi",
            model_name="model",
            score=0.8,
            metrics={
                "initial_performance": 0.6,
                "final_performance": 0.8,
                "total_improvements": 3,
                "safety_violations": 0
            }
        )

        assert benchmark.validate_result(valid_result)

        # Test invalid result
        invalid_result = BenchmarkResult(
            benchmark_name="test",
            benchmark_category="asi",
            model_name="model",
            score=0.8,
            metrics={}  # Missing required metrics
        )

        assert not benchmark.validate_result(invalid_result)

    def test_cleanup(self, benchmark):
        """Test benchmark cleanup."""
        benchmark.setup()
        benchmark.cleanup()

        assert len(benchmark.performance_history) == 0
        assert len(benchmark.test_problems) == 0


class TestSafetyMonitor:
    """Test safety monitoring functionality."""

    @pytest.fixture
    def safety_monitor(self):
        """Create safety monitor instance."""
        return SafetyMonitor(threshold=0.9)

    def test_initialization(self, safety_monitor):
        """Test safety monitor initialization."""
        assert safety_monitor.threshold == 0.9
        assert len(safety_monitor.violation_history) == 0

    def test_check_safety_high_risk(self, safety_monitor):
        """Test safety check for high-risk improvements."""
        improvement = {
            "type": "architectural",
            "risk_level": "high",
            "description": "High-risk modification"
        }

        # High-risk improvements should be rejected
        assert not safety_monitor.check_safety(None, improvement)

    def test_check_safety_medium_risk(self, safety_monitor):
        """Test safety check for medium-risk improvements."""
        improvement = {
            "type": "parameter",
            "risk_level": "medium",
            "description": "Medium-risk modification"
        }

        # Medium-risk improvements should be allowed
        assert safety_monitor.check_safety(None, improvement)

    def test_check_safety_low_risk(self, safety_monitor):
        """Test safety check for low-risk improvements."""
        improvement = {
            "type": "optimization",
            "risk_level": "low",
            "description": "Low-risk modification"
        }

        # Low-risk improvements should be allowed
        assert safety_monitor.check_safety(None, improvement)


class TestUniversalProblemSolverBenchmark:
    """Test universal problem solver benchmark."""

    @pytest.fixture
    def benchmark(self, asi_config):
        """Create universal problem solver benchmark."""
        return UniversalProblemSolverBenchmark(asi_config)

    def test_initialization(self, benchmark, asi_config):
        """Test benchmark initialization."""
        assert benchmark.config == asi_config
        assert benchmark.max_problems_per_domain == 2
        assert benchmark.time_limit_per_problem == 30

    def test_setup(self, benchmark):
        """Test benchmark setup."""
        benchmark.setup()

        assert hasattr(benchmark, 'domains')
        assert hasattr(benchmark, 'evaluators')
        assert hasattr(benchmark, 'solution_history')
        assert hasattr(benchmark, 'problem_attempts')

        assert len(benchmark.domains) > 0
        assert len(benchmark.evaluators) > 0

    def test_initialize_problem_domains(self, benchmark):
        """Test problem domain initialization."""
        domains = benchmark._initialize_problem_domains()

        assert isinstance(domains, list)
        assert len(domains) > 0

        for domain in domains:
            required_attrs = ["name", "complexity", "problem_types",
                            "constraints", "evaluation_criteria"]
            for attr in required_attrs:
                assert hasattr(domain, attr)

    def test_initialize_evaluators(self, benchmark):
        """Test evaluator initialization."""
        evaluators = benchmark._initialize_evaluators()

        assert isinstance(evaluators, dict)
        assert len(evaluators) > 0

        # Check that evaluators are properly instantiated
        for domain_name, evaluator in evaluators.items():
            assert hasattr(evaluator, 'validate')

    def test_generate_domain_problems(self, benchmark):
        """Test domain problem generation."""
        from ai_benchmarks.benchmarks.asi.universal_problem_solver import ProblemDomain

        domain = ProblemDomain(
            name="test_domain",
            complexity="medium",
            problem_types=["test_type"],
            constraints={},
            evaluation_criteria=["accuracy"]
        )

        problems = benchmark._generate_domain_problems(domain)

        assert isinstance(problems, list)
        assert len(problems) == benchmark.max_problems_per_domain

        for problem in problems:
            assert "domain" in problem
            assert "type" in problem
            assert "difficulty" in problem

    def test_solve_problem(self, benchmark, mock_model):
        """Test problem solving."""
        problem = {
            "type": "test_problem",
            "domain": "test",
            "difficulty": "medium"
        }

        solution = benchmark._solve_problem(mock_model, problem, benchmark.domains[0])

        assert isinstance(solution, dict)
        assert "solution" in solution
        assert "confidence" in solution
        assert "reasoning_steps" in solution

    def test_validate_solution(self, benchmark):
        """Test solution validation."""
        solution = {
            "solution": "test_solution",
            "confidence": 0.8,
            "methodology": "test_method"
        }

        problem = {"type": "test"}
        domain = benchmark.domains[0]

        # Should work with any evaluator
        is_valid = benchmark._validate_solution(solution, problem, domain)
        assert isinstance(is_valid, bool)

    def test_calculate_universal_metrics(self, benchmark):
        """Test universal metrics calculation."""
        domain_results = {
            "mathematical_reasoning": [{"success": True}, {"success": False}],
            "logical_reasoning": [{"success": True}, {"success": True}]
        }

        metrics = benchmark._calculate_universal_metrics(domain_results)

        assert hasattr(metrics, 'domains_solved')
        assert hasattr(metrics, 'total_domains')
        assert hasattr(metrics, 'solution_generality')
        assert 0 <= metrics.solution_generality <= 1

    def test_calculate_overall_score(self, benchmark):
        """Test overall score calculation."""
        from ai_benchmarks.benchmarks.asi.universal_problem_solver import UniversalSolvingMetrics

        metrics = UniversalSolvingMetrics(
            domains_solved=2,
            total_domains=3,
            solution_generality=0.75,
            problem_formulation_accuracy=0.8,
            cross_domain_transfer=0.7,
            meta_reasoning_score=0.85,
            algorithm_discovery_rate=0.6
        )

        score = benchmark._calculate_overall_score(metrics, 0.7, 0.8)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_cleanup(self, benchmark):
        """Test benchmark cleanup."""
        benchmark.setup()
        benchmark.cleanup()

        assert len(benchmark.solution_history) == 0
        assert len(benchmark.problem_attempts) == 0


class TestDomainEvaluators:
    """Test domain-specific evaluators."""

    def test_mathematical_evaluator(self):
        """Test mathematical evaluator."""
        evaluator = MathematicalEvaluator()

        solution = {"confidence": 0.8}
        problem = {"type": "optimization"}

        result = evaluator.validate(solution, problem)
        assert isinstance(result, bool)

    def test_logical_evaluator(self):
        """Test logical evaluator."""
        evaluator = LogicalEvaluator()

        solution = {"confidence": 0.85}
        problem = {"type": "theorem_proving"}

        result = evaluator.validate(solution, problem)
        assert isinstance(result, bool)

    def test_scientific_evaluator(self):
        """Test scientific evaluator."""
        evaluator = ScientificEvaluator()

        solution = {"confidence": 0.9}
        problem = {"type": "hypothesis_generation"}

        result = evaluator.validate(solution, problem)
        assert isinstance(result, bool)

    def test_strategic_evaluator(self):
        """Test strategic evaluator."""
        evaluator = StrategicEvaluator()

        solution = {"confidence": 0.75}
        problem = {"type": "game_theory"}

        result = evaluator.validate(solution, problem)
        assert isinstance(result, bool)

    def test_creative_evaluator(self):
        """Test creative evaluator."""
        evaluator = CreativeEvaluator()

        solution = {"confidence": 0.6}
        problem = {"type": "innovation"}

        result = evaluator.validate(solution, problem)
        assert isinstance(result, bool)
