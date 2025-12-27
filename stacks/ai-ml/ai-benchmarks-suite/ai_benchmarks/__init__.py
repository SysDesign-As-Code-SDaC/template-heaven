"""
Comprehensive AI Benchmarks Suite

A unified framework for benchmarking AI systems across all intelligence levels
and computational paradigms, from Artificial Super Intelligence to neuromorphic computing.

This suite provides standardized evaluation frameworks for:
- Artificial Super Intelligence (ASI) benchmarks
- Artificial General Intelligence (AGI) benchmarks
- Neuromorphic computing benchmarks
- Hybrid LLM architectures
- Quantum AI systems
- Swarm intelligence
- Embodied AI
- Causal reasoning
- Multi-modal learning
- Continual learning
- Adversarial robustness
- Interpretability and explainability

Main Classes:
    BenchmarkRunner: Main execution engine for benchmark suites
    BenchmarkSuiteBuilder: Fluent API for creating custom benchmark suites

Quick Start:
    >>> from ai_benchmarks import BenchmarkRunner, create_comprehensive_ai_suite
    >>> suite = create_comprehensive_ai_suite()
    >>> runner = BenchmarkRunner(suite)
    >>> results = await runner.run_benchmark_suite(your_model)

Command Line:
    $ python -m ai_benchmarks run --suite comprehensive
    $ python -m ai_benchmarks list
    $ python -m ai_benchmarks report --input results/

Author: Template Heaven Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Template Heaven Team"
__description__ = "Comprehensive AI Benchmarks Suite for all intelligence paradigms"

# Main API exports
from .core.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSuiteBuilder,
    create_comprehensive_ai_suite,
    create_quick_evaluation_suite,
    create_specialized_suite,
    BenchmarkSuiteResult
)

from .core.base_benchmark import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRegistry,
    benchmark_registry
)

# Benchmark categories
from . import benchmarks

# Utility functions
def get_available_benchmarks(category=None):
    """
    Get list of available benchmarks.

    Args:
        category: Optional category filter

    Returns:
        Dict mapping categories to benchmark lists
    """
    return BenchmarkRegistry.list_benchmarks(category)


def create_custom_suite(name, categories=None, benchmarks=None, **kwargs):
    """
    Create a custom benchmark suite.

    Args:
        name: Suite name
        categories: List of benchmark categories to include
        benchmarks: List of specific benchmarks to include
        **kwargs: Additional suite configuration

    Returns:
        BenchmarkSuiteConfig
    """
    builder = BenchmarkSuiteBuilder(name)

    if categories:
        builder.include_categories(*categories)

    if benchmarks:
        builder.include_benchmarks(*benchmarks)

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(builder, key):
            getattr(builder, key)(value)
        else:
            builder.custom_parameters({key: value})

    return builder.build()


# Convenience functions for common use cases
async def benchmark_model(model, suite_type="comprehensive", **kwargs):
    """
    Benchmark an AI model with a predefined suite.

    Args:
        model: The AI model to benchmark
        suite_type: Type of suite ("comprehensive", "quick", or "custom")
        **kwargs: Additional configuration options

    Returns:
        BenchmarkSuiteResult
    """
    if suite_type == "comprehensive":
        config = create_comprehensive_ai_suite()
    elif suite_type == "quick":
        config = create_quick_evaluation_suite()
    else:
        # Custom suite
        config = create_custom_suite(f"Custom {suite_type}", **kwargs)

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    runner = BenchmarkRunner(config)
    return await runner.run_benchmark_suite(model)


def run_cli():
    """Run the command-line interface."""
    import sys
    from .cli import main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


__all__ = [
    # Main classes
    "BenchmarkRunner",
    "BenchmarkSuiteBuilder",
    "BaseBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRegistry",
    "BenchmarkSuiteResult",

    # Suite creation functions
    "create_comprehensive_ai_suite",
    "create_quick_evaluation_suite",
    "create_specialized_suite",
    "create_custom_suite",

    # Utility functions
    "get_available_benchmarks",
    "benchmark_model",

    # CLI
    "run_cli",

    # Benchmark modules
    "benchmarks"
]
