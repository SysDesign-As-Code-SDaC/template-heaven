#!/usr/bin/env python3
"""
Basic Usage Example for AI Benchmarks Suite

This example demonstrates how to use the Comprehensive AI Benchmarks Suite
to evaluate AI models across multiple intelligence paradigms.
"""

import asyncio
import logging
from pathlib import Path

# Import the benchmark suite
from ai_benchmarks import (
    BenchmarkRunner,
    create_comprehensive_ai_suite,
    create_quick_evaluation_suite,
    BenchmarkSuiteBuilder
)


class ExampleAIModel:
    """Example AI model for demonstration purposes."""

    def __init__(self, name: str = "example_model"):
        self.name = name
        self.capabilities = {
            'recursive_improvement': 0.7,
            'universal_reasoning': 0.6,
            'cognitive_flexibility': 0.8,
            'temporal_processing': 0.5,
            'ensemble_integration': 0.9
        }

    def __str__(self):
        return f"ExampleAIModel(name='{self.name}')"


async def run_comprehensive_benchmark():
    """Run the comprehensive AI benchmark suite."""
    print("🚀 Running Comprehensive AI Benchmark Suite")
    print("=" * 50)

    # Create the comprehensive benchmark suite
    suite_config = create_comprehensive_ai_suite()

    # Initialize model
    model = ExampleAIModel("comprehensive_test_model")

    # Create benchmark runner
    runner = BenchmarkRunner(suite_config)

    # Run benchmarks
    results = await runner.run_benchmark_suite(model)

    # Print results summary
    print("
📊 Benchmark Results Summary:"    print(f"Total Benchmarks: {results.total_benchmarks}")
    print(f"Successful: {results.successful_benchmarks}")
    print(f"Failed: {results.failed_benchmarks}")
    print(".3f"    print(".3f"
    # Show category breakdown
    if 'category_performance' in results.summary:
        print("\n🏆 Category Performance:")
        for category, perf in results.summary['category_performance'].items():
            avg_score = perf.get('average_score', 0)
            completed = perf.get('completed_benchmarks', 0)
            total = perf.get('total_benchmarks', 0)
            print(".3f"
    return results


async def run_quick_evaluation():
    """Run a quick evaluation benchmark."""
    print("\n⚡ Running Quick Evaluation Suite")
    print("=" * 40)

    # Create quick evaluation suite
    suite_config = create_quick_evaluation_suite()

    # Initialize model
    model = ExampleAIModel("quick_eval_model")

    # Create and run
    runner = BenchmarkRunner(suite_config)
    results = await runner.run_benchmark_suite(model)

    print("
📈 Quick Evaluation Results:"    print(f"Benchmarks: {results.successful_benchmarks}/{results.total_benchmarks}")
    print(".3f"    print(".3f"
    return results


async def run_custom_benchmark():
    """Run a custom benchmark configuration."""
    print("\n🔧 Running Custom Benchmark Suite")
    print("=" * 40)

    # Create custom suite with specific categories
    custom_suite = (BenchmarkSuiteBuilder("Custom Intelligence Suite")
        .description("Focused evaluation on core intelligence capabilities")
        .include_categories("agi", "neuromorphic")
        .include_benchmarks("general_intelligence", "spiking_neural_networks")
        .parallel_execution(False)  # Sequential for this example
        .timeout(300)  # 5 minutes per benchmark
        .output_directory("custom_results")
        .build())

    # Initialize model
    model = ExampleAIModel("custom_model")

    # Create and run
    runner = BenchmarkRunner(custom_suite)
    results = await runner.run_benchmark_suite(model)

    print("
🛠️ Custom Suite Results:"    print(f"Benchmarks: {results.successful_benchmarks}/{results.total_benchmarks}")
    print(".3f"
    return results


async def demonstrate_benchmark_categories():
    """Demonstrate different benchmark categories."""
    print("\n📚 Available Benchmark Categories:")
    print("=" * 40)

    from ai_benchmarks.core.base_benchmark import BenchmarkRegistry

    categories = BenchmarkRegistry.list_categories()
    all_benchmarks = BenchmarkRegistry.list_benchmarks()

    for category in categories:
        print(f"\n🤖 {category.upper()}:")
        if category in all_benchmarks:
            for benchmark_name in all_benchmarks[category]:
                print(f"  • {benchmark_name}")
        else:
            print("  No benchmarks available")


async def main():
    """Main demonstration function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🎯 AI Benchmarks Suite Demonstration")
    print("This example shows how to evaluate AI models across multiple intelligence paradigms\n")

    try:
        # Demonstrate available categories
        await demonstrate_benchmark_categories()

        # Run different benchmark suites
        comprehensive_results = await run_comprehensive_benchmark()
        quick_results = await run_quick_evaluation()
        custom_results = await run_custom_benchmark()

        # Summary
        print("\n🎉 Benchmark Demonstration Complete!")
        print("=" * 50)
        print("Summary of all benchmark runs:")
        print(f"Comprehensive: {comprehensive_results.successful_benchmarks}/{comprehensive_results.total_benchmarks} successful")
        print(f"Quick Eval:    {quick_results.successful_benchmarks}/{quick_results.total_benchmarks} successful")
        print(f"Custom:        {custom_results.successful_benchmarks}/{custom_results.total_benchmarks} successful")

        print("
📁 Results saved to:"        print("  - benchmark_results/ (comprehensive)")
        print("  - custom_results/ (custom)")
        print("\n📊 Check the generated HTML reports for detailed visualizations!")

    except Exception as e:
        print(f"❌ Error during benchmark execution: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(0)
