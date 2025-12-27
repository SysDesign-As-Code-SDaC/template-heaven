"""
Command Line Interface for AI Benchmarks Suite

This module provides a command-line interface for running comprehensive
AI benchmarks across all intelligence paradigms.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .core.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkSuiteBuilder,
    create_comprehensive_ai_suite,
    create_quick_evaluation_suite,
    create_specialized_suite
)
from ..utils.config_loader import ConfigLoader
from ..utils.reporting import BenchmarkReporter


class BenchmarkCLI:
    """Command-line interface for AI benchmarks."""

    def __init__(self):
        self.config_loader = ConfigLoader()
        self.reporter = None

    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="Comprehensive AI Benchmarks Suite",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run comprehensive benchmark suite
  python -m ai_benchmarks run --suite comprehensive

  # Run specific categories
  python -m ai_benchmarks run --categories asi agi neuromorphic

  # Run specific benchmarks
  python -m ai_benchmarks run --benchmarks recursive_self_improvement general_intelligence

  # Generate reports
  python -m ai_benchmarks report --input results/ --output reports/

  # List available benchmarks
  python -m ai_benchmarks list
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Run command
        run_parser = subparsers.add_parser('run', help='Run benchmark suite')
        run_parser.add_argument('--suite', choices=['comprehensive', 'quick', 'custom'],
                               default='comprehensive', help='Predefined benchmark suite')
        run_parser.add_argument('--categories', nargs='+',
                               choices=['asi', 'agi', 'neuromorphic', 'hybrid_llms', 'quantum_ai',
                                       'swarm_intelligence', 'embodied_ai', 'causal_reasoning',
                                       'multi_modal', 'continual_learning', 'adversarial_robustness',
                                       'interpretability'],
                               help='Benchmark categories to run')
        run_parser.add_argument('--benchmarks', nargs='+', help='Specific benchmarks to run')
        run_parser.add_argument('--config', type=str, help='Custom configuration file')
        run_parser.add_argument('--output', type=str, default='benchmark_results',
                               help='Output directory for results')
        run_parser.add_argument('--model', type=str, help='Model identifier or path')
        run_parser.add_argument('--parallel', action='store_true', default=True,
                               help='Enable parallel execution')
        run_parser.add_argument('--workers', type=int, default=4,
                               help='Number of parallel workers')

        # Report command
        report_parser = subparsers.add_parser('report', help='Generate benchmark reports')
        report_parser.add_argument('--input', type=str, required=True,
                                  help='Input directory with benchmark results')
        report_parser.add_argument('--output', type=str, default='reports',
                                  help='Output directory for reports')
        report_parser.add_argument('--format', choices=['html', 'json', 'csv', 'pdf'],
                                  default='html', help='Report format')

        # List command
        list_parser = subparsers.add_parser('list', help='List available benchmarks')
        list_parser.add_argument('--category', choices=['asi', 'agi', 'neuromorphic', 'hybrid_llms'],
                                help='Filter by category')

        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate benchmark results')
        validate_parser.add_argument('--input', type=str, required=True,
                                    help='Input directory with benchmark results')
        validate_parser.add_argument('--output', type=str, help='Output file for validation report')

        return parser

    def load_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Load configuration based on command-line arguments."""
        if args.config:
            return self.config_loader.load_config(args.config)
        else:
            return self.config_loader.load_default_config()

    def create_suite_config(self, args: argparse.Namespace) -> Any:
        """Create benchmark suite configuration from arguments."""
        if args.suite == 'comprehensive':
            return create_comprehensive_ai_suite()
        elif args.suite == 'quick':
            return create_quick_evaluation_suite()
        elif args.suite == 'custom':
            builder = BenchmarkSuiteBuilder("Custom Suite")

            if args.categories:
                builder.include_categories(*args.categories)

            if args.benchmarks:
                builder.include_benchmarks(*args.benchmarks)

            builder.parallel_execution(args.parallel, args.workers)
            builder.output_directory(args.output)

            return builder.build()
        else:
            # Specialized suite
            if args.categories and len(args.categories) == 1:
                return create_specialized_suite(args.categories[0])
            else:
                return create_comprehensive_ai_suite()

    async def run_benchmarks(self, args: argparse.Namespace) -> int:
        """Run benchmark suite."""
        try:
            # Create suite configuration
            suite_config = self.create_suite_config(args)

            # Initialize reporter
            self.reporter = BenchmarkReporter(suite_config.output_directory)

            # Create mock model for demonstration
            # In real usage, this would load an actual AI model
            model = MockAIModel(args.model or "demo_model")

            # Create and run benchmark runner
            runner = BenchmarkRunner(suite_config)

            print(f"Starting benchmark suite: {suite_config.name}")
            print(f"Description: {suite_config.description}")
            print(f"Categories: {', '.join(suite_config.benchmark_categories)}")
            print(f"Parallel execution: {suite_config.parallel_execution}")
            print("-" * 50)

            # Run benchmarks
            results = await runner.run_benchmark_suite(model)

            # Print summary
            print("\nBenchmark Suite Results:")
            print(f"Total Benchmarks: {results.total_benchmarks}")
            print(f"Successful: {results.successful_benchmarks}")
            print(f"Failed: {results.failed_benchmarks}")
            print(".2f"            print(".2f"            print(f"Execution Time: {results.execution_time:.2f} seconds")

            if results.summary.get('category_performance'):
                print("\nCategory Performance:")
                for category, perf in results.summary['category_performance'].items():
                    avg_score = perf.get('average_score', 0)
                    completed = perf.get('completed_benchmarks', 0)
                    total = perf.get('total_benchmarks', 0)
                    print(f"  {category}: {avg_score:.3f} ({completed}/{total} completed)")

            # Save detailed results
            results_file = Path(suite_config.output_directory) / "suite_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)

            with open(results_file, 'w') as f:
                json.dump({
                    'suite_info': {
                        'name': results.suite_name,
                        'start_time': results.start_time.isoformat(),
                        'end_time': results.end_time.isoformat(),
                        'execution_time': results.execution_time
                    },
                    'summary': results.summary,
                    'results': [r.to_dict() for r in results.results]
                }, f, indent=2, default=str)

            print(f"\nDetailed results saved to: {results_file}")

            # Generate reports if requested
            if suite_config.generate_reports:
                print("\nGenerating reports...")
                await self.reporter.generate_html_report(
                    results.results, results.summary, results.suite_name
                )
                print("Reports generated successfully")

            return 0 if results.successful_benchmarks > 0 else 1

        except Exception as e:
            print(f"Error running benchmarks: {str(e)}", file=sys.stderr)
            return 1

    async def generate_reports(self, args: argparse.Namespace) -> int:
        """Generate benchmark reports."""
        try:
            input_dir = Path(args.input)
            if not input_dir.exists():
                print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
                return 1

            # Load results
            results_file = input_dir / "suite_results.json"
            if not results_file.exists():
                print(f"Results file not found: {results_file}", file=sys.stderr)
                return 1

            with open(results_file, 'r') as f:
                data = json.load(f)

            # Create reporter
            reporter = BenchmarkReporter(args.output)

            # Generate reports
            if args.format == 'html':
                await reporter.generate_html_report(
                    data['results'], data['summary'], data['suite_info']['name']
                )
            elif args.format == 'json':
                await reporter.generate_json_summary(
                    data['results'], data['summary'], data['suite_info']['name']
                )

            print(f"Reports generated in: {args.output}")
            return 0

        except Exception as e:
            print(f"Error generating reports: {str(e)}", file=sys.stderr)
            return 1

    def list_benchmarks(self, args: argparse.Namespace) -> int:
        """List available benchmarks."""
        from .core.base_benchmark import BenchmarkRegistry

        print("Available AI Benchmarks:")
        print("=" * 50)

        categories = BenchmarkRegistry.list_categories()
        all_benchmarks = BenchmarkRegistry.list_benchmarks()

        for category in categories:
            if args.category and category != args.category:
                continue

            print(f"\n{category.upper()} Benchmarks:")
            print("-" * 30)

            if category in all_benchmarks:
                for benchmark_name in all_benchmarks[category]:
                    benchmark_class = BenchmarkRegistry.get_benchmark(category, benchmark_name)
                    if benchmark_class:
                        description = getattr(benchmark_class, '__doc__', 'No description').strip().split('\n')[0]
                        print(f"  • {benchmark_name}: {description}")
            else:
                print("  No benchmarks available")

        return 0

    async def validate_results(self, args: argparse.Namespace) -> int:
        """Validate benchmark results."""
        try:
            input_dir = Path(args.input)
            if not input_dir.exists():
                print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
                return 1

            # Load results
            results_file = input_dir / "suite_results.json"
            if not results_file.exists():
                print(f"Results file not found: {results_file}", file=sys.stderr)
                return 1

            with open(results_file, 'r') as f:
                data = json.load(f)

            # Perform validation
            validation_results = self._validate_benchmark_results(data['results'])

            # Save validation report
            output_file = args.output or str(input_dir / "validation_report.json")
            with open(output_file, 'w') as f:
                json.dump(validation_results, f, indent=2)

            print(f"Validation report saved to: {output_file}")

            # Print summary
            valid_results = sum(1 for r in validation_results['validations'] if r['is_valid'])
            total_results = len(validation_results['validations'])

            print(f"Validation Summary: {valid_results}/{total_results} results are valid")

            return 0 if valid_results == total_results else 1

        except Exception as e:
            print(f"Error validating results: {str(e)}", file=sys.stderr)
            return 1

    def _validate_benchmark_results(self, results: list) -> Dict[str, Any]:
        """Validate benchmark results for consistency and correctness."""
        validations = []

        for result in results:
            validation = {
                'benchmark_name': result.get('benchmark_name'),
                'is_valid': True,
                'issues': []
            }

            # Check required fields
            required_fields = ['benchmark_name', 'benchmark_category', 'score', 'status']
            for field in required_fields:
                if field not in result:
                    validation['issues'].append(f"Missing required field: {field}")
                    validation['is_valid'] = False

            # Validate score range
            score = result.get('score')
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                validation['issues'].append(f"Invalid score: {score}")
                validation['is_valid'] = False

            # Validate status
            valid_statuses = ['completed', 'failed', 'validation_failed']
            status = result.get('status')
            if status not in valid_statuses:
                validation['issues'].append(f"Invalid status: {status}")
                validation['is_valid'] = False

            validations.append(validation)

        return {
            'total_results': len(results),
            'valid_results': sum(1 for v in validations if v['is_valid']),
            'validations': validations
        }

    async def run(self, args: argparse.Namespace) -> int:
        """Run the appropriate command."""
        if args.command == 'run':
            return await self.run_benchmarks(args)
        elif args.command == 'report':
            return await self.generate_reports(args)
        elif args.command == 'list':
            return self.list_benchmarks(args)
        elif args.command == 'validate':
            return await self.validate_results(args)
        else:
            print("No command specified. Use --help for usage information.", file=sys.stderr)
            return 1


class MockAIModel:
    """Mock AI model for demonstration purposes."""

    def __init__(self, name: str = "mock_model"):
        self.name = name

    def __str__(self):
        return f"MockAIModel(name='{self.name}')"


def main():
    """Main entry point for CLI."""
    cli = BenchmarkCLI()
    parser = cli.create_parser()
    args = parser.parse_args()

    # Run async command
    exit_code = asyncio.run(cli.run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
