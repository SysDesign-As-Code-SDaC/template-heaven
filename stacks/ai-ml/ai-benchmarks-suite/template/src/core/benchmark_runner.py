"""
Unified Benchmark Execution Framework

This module provides the main framework for running comprehensive AI benchmarks
across all intelligence paradigms and computational approaches.
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base_benchmark import BenchmarkRegistry, BenchmarkConfig, BenchmarkResult
from ..utils.reporting import BenchmarkReporter
from ..utils.config_loader import ConfigLoader


@dataclass
class BenchmarkSuiteConfig:
    """Configuration for a benchmark suite run."""

    name: str
    description: str
    benchmark_categories: List[str]
    benchmark_names: Optional[List[str]] = None
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_per_benchmark: float = 3600  # 1 hour
    output_directory: str = "benchmark_results"
    save_intermediate_results: bool = True
    generate_reports: bool = True
    custom_parameters: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkSuiteResult:
    """Results from running a complete benchmark suite."""

    suite_name: str
    start_time: datetime
    end_time: datetime
    total_benchmarks: int
    successful_benchmarks: int
    failed_benchmarks: int
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    execution_time: float


class BenchmarkRunner:
    """
    Main benchmark execution engine.

    This class orchestrates the execution of comprehensive AI benchmarks across
    all intelligence paradigms, providing unified execution, result collection,
    and reporting capabilities.
    """

    def __init__(self, config: BenchmarkSuiteConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

        # Initialize components
        self.reporter = BenchmarkReporter(config.output_directory)
        self.config_loader = ConfigLoader()

        # Execution tracking
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers) if config.parallel_execution else None

    def _setup_logging(self):
        """Setup comprehensive logging for benchmark execution."""
        self.logger.setLevel(logging.INFO)

        # Create logs directory
        log_dir = Path(self.config.output_directory) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        log_file = log_dir / f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    async def run_benchmark_suite(self, model: Any) -> BenchmarkSuiteResult:
        """
        Execute a complete benchmark suite.

        Args:
            model: The AI model to benchmark

        Returns:
            Comprehensive benchmark suite results
        """
        self.logger.info(f"Starting benchmark suite: {self.config.name}")
        start_time = datetime.utcnow()

        try:
            # Load benchmark configurations
            benchmark_configs = self._load_benchmark_configs()

            # Execute benchmarks
            if self.config.parallel_execution:
                results = await self._run_benchmarks_parallel(benchmark_configs, model)
            else:
                results = await self._run_benchmarks_sequential(benchmark_configs, model)

            # Calculate suite summary
            summary = self._calculate_suite_summary(results)

            # Generate reports
            if self.config.generate_reports:
                await self._generate_reports(results, summary)

            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            suite_result = BenchmarkSuiteResult(
                suite_name=self.config.name,
                start_time=start_time,
                end_time=end_time,
                total_benchmarks=len(benchmark_configs),
                successful_benchmarks=len([r for r in results if r.status == "completed"]),
                failed_benchmarks=len([r for r in results if r.status != "completed"]),
                results=results,
                summary=summary,
                execution_time=execution_time
            )

            self.logger.info(f"Benchmark suite completed: {suite_result.successful_benchmarks}/{suite_result.total_benchmarks} successful")

            return suite_result

        except Exception as e:
            self.logger.error(f"Benchmark suite execution failed: {str(e)}")
            raise

    def _load_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Load configurations for all benchmarks to run."""
        configs = []

        for category in self.config.benchmark_categories:
            if self.config.benchmark_names:
                # Run specific benchmarks
                for benchmark_name in self.config.benchmark_names:
                    benchmark_class = BenchmarkRegistry.get_benchmark(category, benchmark_name)
                    if benchmark_class:
                        config = self._create_benchmark_config(category, benchmark_name)
                        configs.append(config)
            else:
                # Run all benchmarks in category
                category_benchmarks = BenchmarkRegistry.list_benchmarks(category)
                if category in category_benchmarks:
                    for benchmark_name in category_benchmarks[category]:
                        config = self._create_benchmark_config(category, benchmark_name)
                        configs.append(config)

        self.logger.info(f"Loaded {len(configs)} benchmark configurations")
        return configs

    def _create_benchmark_config(self, category: str, name: str) -> BenchmarkConfig:
        """Create configuration for a specific benchmark."""
        # Load base configuration
        base_config = {
            "name": f"{category}_{name}",
            "category": category,
            "description": f"Benchmark for {name} in {category} category",
            "timeout": self.config.timeout_per_benchmark
        }

        # Add custom parameters if provided
        if self.config.custom_parameters and name in self.config.custom_parameters:
            base_config["parameters"] = self.config.custom_parameters[name]

        return BenchmarkConfig(**base_config)

    async def _run_benchmarks_parallel(self, configs: List[BenchmarkConfig], model: Any) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        self.logger.info(f"Running {len(configs)} benchmarks in parallel")

        # Create tasks
        tasks = []
        for config in configs:
            task = self._run_single_benchmark_async(config, model)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and convert to results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed benchmark
                error_result = BenchmarkResult(
                    benchmark_name=configs[i].name,
                    benchmark_category=configs[i].category,
                    model_name=getattr(model, 'name', 'unknown'),
                    score=0.0,
                    status="failed",
                    metadata={"error": str(result)}
                )
                processed_results.append(error_result)
                self.logger.error(f"Benchmark {configs[i].name} failed: {str(result)}")
            else:
                processed_results.append(result)

        return processed_results

    async def _run_benchmarks_sequential(self, configs: List[BenchmarkConfig], model: Any) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        self.logger.info(f"Running {len(configs)} benchmarks sequentially")

        results = []
        for config in configs:
            try:
                result = await self._run_single_benchmark_async(config, model)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = BenchmarkResult(
                    benchmark_name=config.name,
                    benchmark_category=config.category,
                    model_name=getattr(model, 'name', 'unknown'),
                    score=0.0,
                    status="failed",
                    metadata={"error": str(e)}
                )
                results.append(error_result)
                self.logger.error(f"Benchmark {config.name} failed: {str(e)}")

        return results

    async def _run_single_benchmark_async(self, config: BenchmarkConfig, model: Any) -> BenchmarkResult:
        """Run a single benchmark asynchronously."""
        def run_benchmark():
            benchmark_class = BenchmarkRegistry.get_benchmark(config.category, config.name.split('_', 1)[1])
            if not benchmark_class:
                raise ValueError(f"Benchmark {config.name} not found")

            benchmark = benchmark_class(config)
            return benchmark.execute(model)

        # Run in thread pool to avoid blocking
        if self.executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, run_benchmark)
        else:
            result = run_benchmark()

        # Save intermediate results if requested
        if self.config.save_intermediate_results:
            await self._save_intermediate_result(result)

        return result

    async def _save_intermediate_result(self, result: BenchmarkResult):
        """Save intermediate benchmark result."""
        try:
            result_file = Path(self.config.output_directory) / "intermediate" / f"{result.benchmark_name}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)

            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

        except Exception as e:
            self.logger.warning(f"Failed to save intermediate result: {str(e)}")

    def _calculate_suite_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate comprehensive suite summary."""
        if not results:
            return {}

        # Basic statistics
        scores = [r.score for r in results if r.status == "completed"]
        categories = list(set(r.benchmark_category for r in results))

        summary = {
            "total_benchmarks": len(results),
            "completed_benchmarks": len([r for r in results if r.status == "completed"]),
            "failed_benchmarks": len([r for r in results if r.status != "completed"]),
            "categories_covered": categories,
            "average_score": np.mean(scores) if scores else 0.0,
            "score_std": np.std(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "score_distribution": self._calculate_score_distribution(scores),
            "category_performance": self._calculate_category_performance(results),
            "benchmark_failures": [r.benchmark_name for r in results if r.status != "completed"]
        }

        return summary

    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate score distribution across ranges."""
        if not scores:
            return {}

        ranges = {
            "excellent": len([s for s in scores if s >= 0.9]),
            "good": len([s for s in scores if 0.7 <= s < 0.9]),
            "fair": len([s for s in scores if 0.5 <= s < 0.7]),
            "poor": len([s for s in scores if s < 0.5])
        }

        return ranges

    def _calculate_category_performance(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance statistics per category."""
        category_stats = {}

        for category in set(r.benchmark_category for r in results):
            category_results = [r for r in results if r.benchmark_category == category]
            completed_results = [r for r in category_results if r.status == "completed"]

            if completed_results:
                scores = [r.score for r in completed_results]
                category_stats[category] = {
                    "total_benchmarks": len(category_results),
                    "completed_benchmarks": len(completed_results),
                    "average_score": np.mean(scores),
                    "best_score": max(scores),
                    "worst_score": min(scores)
                }
            else:
                category_stats[category] = {
                    "total_benchmarks": len(category_results),
                    "completed_benchmarks": 0,
                    "average_score": 0.0,
                    "best_score": 0.0,
                    "worst_score": 0.0
                }

        return category_stats

    async def _generate_reports(self, results: List[BenchmarkResult], summary: Dict[str, Any]):
        """Generate comprehensive benchmark reports."""
        try:
            # Generate HTML report
            await self.reporter.generate_html_report(results, summary, self.config.name)

            # Generate JSON summary
            await self.reporter.generate_json_summary(results, summary, self.config.name)

            # Generate performance comparison charts
            await self.reporter.generate_performance_charts(results)

            # Generate category analysis
            await self.reporter.generate_category_analysis(results)

            self.logger.info("Benchmark reports generated successfully")

        except Exception as e:
            self.logger.error(f"Failed to generate reports: {str(e)}")

    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

        self.logger.info("Benchmark runner cleanup completed")


class BenchmarkSuiteBuilder:
    """
    Builder for creating benchmark suite configurations.

    This class provides a fluent interface for building comprehensive
    benchmark suites with various configuration options.
    """

    def __init__(self, name: str):
        self.config = BenchmarkSuiteConfig(
            name=name,
            description="",
            benchmark_categories=[]
        )

    def description(self, desc: str) -> 'BenchmarkSuiteBuilder':
        """Set suite description."""
        self.config.description = desc
        return self

    def include_categories(self, *categories: str) -> 'BenchmarkSuiteBuilder':
        """Include benchmark categories."""
        self.config.benchmark_categories.extend(categories)
        return self

    def include_benchmarks(self, *benchmark_names: str) -> 'BenchmarkSuiteBuilder':
        """Include specific benchmarks."""
        if not self.config.benchmark_names:
            self.config.benchmark_names = []
        self.config.benchmark_names.extend(benchmark_names)
        return self

    def parallel_execution(self, enabled: bool = True, max_workers: int = 4) -> 'BenchmarkSuiteBuilder':
        """Configure parallel execution."""
        self.config.parallel_execution = enabled
        self.config.max_workers = max_workers
        return self

    def timeout(self, timeout_seconds: float) -> 'BenchmarkSuiteBuilder':
        """Set timeout per benchmark."""
        self.config.timeout_per_benchmark = timeout_seconds
        return self

    def output_directory(self, directory: str) -> 'BenchmarkSuiteBuilder':
        """Set output directory for results."""
        self.config.output_directory = directory
        return self

    def save_intermediates(self, save: bool = True) -> 'BenchmarkSuiteBuilder':
        """Configure intermediate result saving."""
        self.config.save_intermediate_results = save
        return self

    def custom_parameters(self, params: Dict[str, Any]) -> 'BenchmarkSuiteBuilder':
        """Set custom parameters for benchmarks."""
        self.config.custom_parameters = params
        return self

    def build(self) -> BenchmarkSuiteConfig:
        """Build the benchmark suite configuration."""
        return self.config


# Convenience functions for common benchmark suites
def create_comprehensive_ai_suite() -> BenchmarkSuiteConfig:
    """Create a comprehensive AI benchmark suite covering all paradigms."""
    return (BenchmarkSuiteBuilder("Comprehensive AI Benchmarks")
            .description("Complete evaluation across ASI, AGI, neuromorphic, and hybrid paradigms")
            .include_categories("asi", "agi", "neuromorphic", "hybrid_llms")
            .parallel_execution(True, 4)
            .timeout(3600)
            .save_intermediates(True)
            .build())


def create_quick_evaluation_suite() -> BenchmarkSuiteConfig:
    """Create a quick evaluation suite for rapid assessment."""
    return (BenchmarkSuiteBuilder("Quick AI Evaluation")
            .description("Fast evaluation of core AI capabilities")
            .include_categories("agi", "hybrid_llms")
            .include_benchmarks("general_intelligence", "hybrid_architecture")
            .parallel_execution(False)
            .timeout(1800)
            .build())


def create_specialized_suite(category: str) -> BenchmarkSuiteConfig:
    """Create a specialized suite for a specific AI paradigm."""
    return (BenchmarkSuiteBuilder(f"Specialized {category.title()} Benchmarks")
            .description(f"Focused evaluation of {category} capabilities")
            .include_categories(category)
            .parallel_execution(True, 2)
            .timeout(2700)
            .save_intermediates(True)
            .build())
