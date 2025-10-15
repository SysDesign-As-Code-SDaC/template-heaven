"""
Base benchmark framework for AI evaluation.

This module provides the foundational classes and interfaces for implementing
comprehensive AI benchmarks across all intelligence paradigms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import numpy as np


@dataclass
class BenchmarkResult:
    """Represents the result of a benchmark evaluation."""

    benchmark_name: str
    benchmark_category: str
    model_name: str
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0
    status: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_category": self.benchmark_category,
            "model_name": self.model_name,
            "score": self.score,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time,
            "status": self.status
        }

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    name: str
    category: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retries: int = 3
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """
    Abstract base class for all AI benchmarks.

    This class provides the interface that all benchmark implementations must follow,
    ensuring consistency across different AI evaluation paradigms.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for the benchmark."""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @abstractmethod
    def setup(self, **kwargs) -> None:
        """
        Setup the benchmark environment.

        This method should prepare any necessary resources, data, or configurations
        required for benchmark execution.
        """
        pass

    @abstractmethod
    def run(self, model: Any, **kwargs) -> BenchmarkResult:
        """
        Execute the benchmark against the given model.

        Args:
            model: The AI model to evaluate
            **kwargs: Additional execution parameters

        Returns:
            BenchmarkResult containing evaluation metrics and scores
        """
        pass

    @abstractmethod
    def validate_result(self, result: BenchmarkResult) -> bool:
        """
        Validate the benchmark result.

        Args:
            result: The benchmark result to validate

        Returns:
            True if result is valid, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources after benchmark execution.

        This method should release any resources allocated during setup or execution.
        """
        pass

    def execute(self, model: Any, **kwargs) -> BenchmarkResult:
        """
        Execute the complete benchmark lifecycle.

        This method orchestrates the full benchmark execution including setup,
        running, validation, and cleanup.

        Args:
            model: The AI model to evaluate
            **kwargs: Additional execution parameters

        Returns:
            BenchmarkResult containing evaluation metrics and scores
        """
        try:
            self.logger.info(f"Starting benchmark: {self.config.name}")

            # Setup phase
            self.setup(**kwargs)

            # Execution phase with timing
            start_time = datetime.utcnow()
            result = self.run(model, **kwargs)
            end_time = datetime.utcnow()

            # Calculate execution time
            result.execution_time = (end_time - start_time).total_seconds()

            # Validation phase
            if self.validate_result(result):
                result.status = "completed"
                self.logger.info(f"Benchmark completed successfully: {result.score:.4f}")
            else:
                result.status = "validation_failed"
                self.logger.warning("Benchmark result validation failed")

            # Cleanup phase
            self.cleanup()

            return result

        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {str(e)}")
            # Create error result
            error_result = BenchmarkResult(
                benchmark_name=self.config.name,
                benchmark_category=self.config.category,
                model_name=getattr(model, 'name', 'unknown'),
                score=0.0,
                status="failed",
                metadata={"error": str(e)}
            )
            self.cleanup()
            return error_result


class BenchmarkRegistry:
    """
    Registry for managing benchmark implementations.

    This class provides a centralized way to register and discover benchmark
    implementations across all AI paradigms.
    """

    _benchmarks: Dict[str, Dict[str, type]] = {}

    @classmethod
    def register(cls, category: str, name: str, benchmark_class: type) -> None:
        """
        Register a benchmark implementation.

        Args:
            category: Benchmark category (e.g., 'asi', 'agi', 'neuromorphic')
            name: Benchmark name
            benchmark_class: The benchmark class to register
        """
        if category not in cls._benchmarks:
            cls._benchmarks[category] = {}

        cls._benchmarks[category][name] = benchmark_class

    @classmethod
    def get_benchmark(cls, category: str, name: str) -> Optional[type]:
        """
        Get a registered benchmark class.

        Args:
            category: Benchmark category
            name: Benchmark name

        Returns:
            The benchmark class if registered, None otherwise
        """
        return cls._benchmarks.get(category, {}).get(name)

    @classmethod
    def list_categories(cls) -> List[str]:
        """List all registered benchmark categories."""
        return list(cls._benchmarks.keys())

    @classmethod
    def list_benchmarks(cls, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered benchmarks.

        Args:
            category: Optional category filter

        Returns:
            Dictionary mapping categories to lists of benchmark names
        """
        if category:
            return {category: list(cls._benchmarks.get(category, {}).keys())}
        else:
            return {cat: list(benchmarks.keys())
                   for cat, benchmarks in cls._benchmarks.items()}


def benchmark_registry(cls):
    """
    Decorator to automatically register benchmark classes.

    Usage:
        @benchmark_registry
        class MyBenchmark(BaseBenchmark):
            CATEGORY = "my_category"
            NAME = "my_benchmark"
    """
    if hasattr(cls, 'CATEGORY') and hasattr(cls, 'NAME'):
        BenchmarkRegistry.register(cls.CATEGORY, cls.NAME, cls)
    return cls
