#!/usr/bin/env python3
"""
AI Benchmarks Suite - Clear Entry Point

This is the main entry point for running comprehensive AI benchmarks
across all intelligence paradigms.

Usage:
    python run_benchmarks.py --help
    python run_benchmarks.py comprehensive
    python run_benchmarks.py quick
    python run_benchmarks.py custom --categories asi agi
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_benchmarks import (
    BenchmarkRunner,
    create_comprehensive_ai_suite,
    create_quick_evaluation_suite,
    create_custom_suite,
    get_available_benchmarks
)


class MockAIModel:
    """Mock AI model for demonstration and testing."""

    def __init__(self, name="demo_model"):
        self.name = name

    def __str__(self):
        return f"MockAIModel(name='{self.name}')"


async def run_comprehensive_benchmarks():
    """Run the comprehensive AI benchmark suite."""
    print("🚀 Running Comprehensive AI Benchmarks Suite")
    print("=" * 60)
    print("This will evaluate your AI model across all intelligence paradigms:")
    print("• ASI (Artificial Super Intelligence)")
    print("• AGI (Artificial General Intelligence)")
    print("• Neuromorphic Computing")
    print("• Hybrid LLMs")
    print("• And more...")
    print()

    # Create comprehensive suite
    config = create_comprehensive_ai_suite()
    runner = BenchmarkRunner(config)

    # Use mock model for demonstration
    model = MockAIModel("comprehensive_test_model")

    print(f"📊 Benchmarking model: {model.name}")
    print(f"🔧 Suite: {config.name}")
    print(f"📈 Categories: {', '.join(config.benchmark_categories)}")
    print("-" * 60)

    try:
        results = await runner.run_benchmark_suite(model)

        print("\n🎉 Benchmark Suite Completed!")
        print("=" * 60)
        print("📊 RESULTS SUMMARY:"        print(f"• Total Benchmarks: {results.total_benchmarks}")
        print(f"• Successful: {results.successful_benchmarks}")
        print(f"• Failed: {results.failed_benchmarks}")
        print(".2f"        print(".3f"        print(f"• Execution Time: {results.execution_time:.2f} seconds")

        if results.summary.get('category_performance'):
            print("\n🏆 CATEGORY PERFORMANCE:")
            for category, perf in results.summary['category_performance'].items():
                avg_score = perf.get('average_score', 0)
                completed = perf.get('completed_benchmarks', 0)
                total = perf.get('total_benchmarks', 0)
                status = "✅" if completed == total else "⚠️"
                print(".3f"
        print("\n📁 Detailed results saved to: benchmark_results/")
        print("📈 Run 'python -m ai_benchmarks report --input benchmark_results/' for detailed reports")

        return results

    except Exception as e:
        print(f"❌ Error during benchmark execution: {str(e)}")
        return None


async def run_quick_benchmarks():
    """Run quick evaluation benchmarks."""
    print("⚡ Running Quick AI Evaluation")
    print("=" * 40)
    print("Fast evaluation across core AI capabilities")
    print()

    config = create_quick_evaluation_suite()
    runner = BenchmarkRunner(config)
    model = MockAIModel("quick_eval_model")

    print(f"📊 Benchmarking model: {model.name}")
    print("-" * 40)

    try:
        results = await runner.run_benchmark_suite(model)

        print("\n✅ Quick Evaluation Completed!")
        print(".2f"        print(f"📁 Results: benchmark_results/")

        return results

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


async def run_custom_benchmarks(categories=None, benchmarks=None):
    """Run custom benchmark configuration."""
    print("🔧 Running Custom AI Benchmarks")
    print("=" * 40)

    if categories:
        print(f"Categories: {', '.join(categories)}")
    if benchmarks:
        print(f"Specific Benchmarks: {', '.join(benchmarks)}")
    print()

    config = create_custom_suite(
        "Custom Benchmarks",
        categories=categories,
        benchmarks=benchmarks
    )

    runner = BenchmarkRunner(config)
    model = MockAIModel("custom_model")

    try:
        results = await runner.run_benchmark_suite(model)

        print("\n✅ Custom Benchmarks Completed!")
        print(".2f"        print(f"📁 Results: benchmark_results/")

        return results

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def list_available_benchmarks():
    """List all available benchmarks."""
    print("📚 Available AI Benchmarks")
    print("=" * 40)

    benchmarks = get_available_benchmarks()

    for category, benchmark_list in benchmarks.items():
        print(f"\n🤖 {category.upper()}:")
        if benchmark_list:
            for benchmark in benchmark_list:
                print(f"  • {benchmark}")
        else:
            print("  No benchmarks available")

    print(f"\n📊 Total Categories: {len(benchmarks)}")


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Benchmarks Suite - Comprehensive AI Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py comprehensive    # Full evaluation
  python run_benchmarks.py quick           # Quick assessment
  python run_benchmarks.py custom --categories asi agi
  python run_benchmarks.py list            # Show available benchmarks

For more options, use: python -m ai_benchmarks --help
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Comprehensive benchmarks
    subparsers.add_parser('comprehensive', help='Run comprehensive AI benchmark suite')

    # Quick benchmarks
    subparsers.add_parser('quick', help='Run quick AI evaluation')

    # Custom benchmarks
    custom_parser = subparsers.add_parser('custom', help='Run custom benchmark configuration')
    custom_parser.add_argument('--categories', nargs='+',
                              choices=['asi', 'agi', 'neuromorphic', 'hybrid_llms'],
                              help='Benchmark categories to include')
    custom_parser.add_argument('--benchmarks', nargs='+',
                              help='Specific benchmarks to run')

    # List benchmarks
    subparsers.add_parser('list', help='List available benchmarks')

    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'comprehensive':
            await run_comprehensive_benchmarks()
        elif args.command == 'quick':
            await run_quick_benchmarks()
        elif args.command == 'custom':
            await run_custom_benchmarks(
                categories=args.categories,
                benchmarks=args.benchmarks
            )
        elif args.command == 'list':
            list_available_benchmarks()
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Benchmark execution interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
