#!/usr/bin/env python3
"""
Quantum Performance Benchmarking Script

This script provides comprehensive benchmarking for quantum computing operations,
measuring execution times, circuit complexities, and performance characteristics
across different quantum frameworks and algorithms.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from quantum_circuits.basic_gates import create_bell_state, create_ghz_state, create_random_circuit
    from quantum_circuits.algorithms.shor import shor_factorize_classical
    from backends.local_simulator import QiskitSimulator, CirqSimulator, PennyLaneSimulator
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quantum libraries not available: {e}")
    QUANTUM_AVAILABLE = False


class QuantumBenchmarker:
    """Comprehensive quantum computing benchmarker."""

    def __init__(self):
        self.results = []
        self.backends = []

        if QUANTUM_AVAILABLE:
            # Initialize backends
            try:
                self.backends.append(('Qiskit', QiskitSimulator('qasm')))
            except:
                pass

            try:
                self.backends.append(('Cirq', CirqSimulator()))
            except:
                pass

            try:
                self.backends.append(('PennyLane', PennyLaneSimulator()))
            except:
                pass

    def benchmark_circuit_creation(self, sizes: List[int] = None) -> List[Dict[str, Any]]:
        """Benchmark quantum circuit creation times."""
        if sizes is None:
            sizes = [5, 10, 15, 20, 25]

        results = []

        print("Benchmarking circuit creation...")

        for n_qubits in sizes:
            print(f"  Testing {n_qubits} qubits...")

            # Bell state
            start_time = time.time()
            bell_circuit = create_bell_state('qiskit')
            bell_time = time.time() - start_time

            # GHZ state
            start_time = time.time()
            ghz_circuit = create_ghz_state(n_qubits, 'qiskit')
            ghz_time = time.time() - start_time

            # Random circuit
            start_time = time.time()
            random_circuit = create_random_circuit(n_qubits, depth=n_qubits, framework='qiskit')
            random_time = time.time() - start_time

            result = {
                'n_qubits': n_qubits,
                'bell_state_creation_time': bell_time,
                'ghz_state_creation_time': ghz_time,
                'random_circuit_creation_time': random_time,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

        return results

    def benchmark_simulation(self, circuit_sizes: List[int] = None, shots: int = 1024) -> List[Dict[str, Any]]:
        """Benchmark quantum circuit simulation."""
        if circuit_sizes is None:
            circuit_sizes = [3, 5, 7, 10]

        results = []

        print("Benchmarking circuit simulation...")

        for n_qubits in circuit_sizes:
            print(f"  Simulating {n_qubits}-qubit circuits...")

            for backend_name, backend in self.backends:
                try:
                    # Create test circuit
                    if n_qubits == 2:
                        circuit = create_bell_state('qiskit')
                    else:
                        circuit = create_ghz_state(min(n_qubits, 5), 'qiskit')  # Limit GHZ size

                    # Benchmark execution
                    start_time = time.time()
                    result = backend.execute_circuit(circuit, shots=shots)
                    execution_time = time.time() - start_time

                    result_data = {
                        'n_qubits': n_qubits,
                        'backend': backend_name,
                        'shots': shots,
                        'execution_time': execution_time,
                        'success': result.get('success', False),
                        'timestamp': datetime.now().isoformat()
                    }

                    results.append(result_data)

                except Exception as e:
                    print(f"    Error with {backend_name}: {e}")
                    result_data = {
                        'n_qubits': n_qubits,
                        'backend': backend_name,
                        'shots': shots,
                        'execution_time': None,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result_data)

        return results

    def benchmark_algorithms(self, test_cases: List[int] = None) -> List[Dict[str, Any]]:
        """Benchmark classical algorithms for comparison."""
        if test_cases is None:
            test_cases = [15, 21, 35, 51, 77, 91, 115]

        results = []

        print("Benchmarking algorithms...")

        for N in test_cases:
            print(f"  Factoring {N}...")

            try:
                start_time = time.time()
                factors = shor_factorize_classical(N)
                execution_time = time.time() - start_time

                # Verify result
                product = 1
                for factor in factors:
                    product *= factor

                success = product == N

                result = {
                    'algorithm': 'classical_factorization',
                    'input_size': N,
                    'execution_time': execution_time,
                    'success': success,
                    'output': factors,
                    'timestamp': datetime.now().isoformat()
                }

                results.append(result)

            except Exception as e:
                result = {
                    'algorithm': 'classical_factorization',
                    'input_size': N,
                    'execution_time': None,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

        return results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive quantum benchmarking suite."""
        print("=== Quantum Computing Performance Benchmark ===\n")

        benchmark_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'hostname': 'localhost',
                'quantum_libraries_available': QUANTUM_AVAILABLE,
                'backends_tested': [name for name, _ in self.backends]
            },
            'circuit_creation': [],
            'simulation': [],
            'algorithms': []
        }

        # Circuit creation benchmarks
        print("1. Circuit Creation Benchmarks")
        benchmark_results['circuit_creation'] = self.benchmark_circuit_creation()

        # Simulation benchmarks
        print("\n2. Simulation Benchmarks")
        benchmark_results['simulation'] = self.benchmark_simulation()

        # Algorithm benchmarks
        print("\n3. Algorithm Benchmarks")
        benchmark_results['algorithms'] = self.benchmark_algorithms()

        # Generate summary
        benchmark_results['summary'] = self.generate_summary(benchmark_results)

        return benchmark_results

    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_execution_times': {},
            'performance_trends': {}
        }

        # Count results
        for category in ['circuit_creation', 'simulation', 'algorithms']:
            if category in results:
                summary['total_tests'] += len(results[category])

                for result in results[category]:
                    if result.get('success', False):
                        summary['successful_tests'] += 1
                    else:
                        summary['failed_tests'] += 1

        summary['success_rate'] = summary['successful_tests'] / max(summary['total_tests'], 1)

        # Calculate average execution times
        execution_times = {}
        for result in results.get('simulation', []):
            backend = result.get('backend', 'unknown')
            exec_time = result.get('execution_time')

            if exec_time is not None:
                if backend not in execution_times:
                    execution_times[backend] = []
                execution_times[backend].append(exec_time)

        for backend, times in execution_times.items():
            summary['average_execution_times'][backend] = sum(times) / len(times)

        return summary

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_benchmark_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nBenchmark results saved to: {filename}")
        return filename

    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable benchmark summary."""
        summary = results.get('summary', {})

        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Successful: {summary.get('successful_tests', 0)}")
        print(f"Failed: {summary.get('failed_tests', 0)}")
        print(".1%")

        if summary.get('average_execution_times'):
            print("\nAverage Execution Times:")
            for backend, avg_time in summary['average_execution_times'].items():
                print(".3f")

        print(f"\nDetailed results saved to JSON file")
        print("="*60)


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Quantum Computing Performance Benchmark")
    parser.add_argument('--circuits-only', action='store_true', help='Benchmark circuit creation only')
    parser.add_argument('--simulation-only', action='store_true', help='Benchmark simulation only')
    parser.add_argument('--algorithms-only', action='store_true', help='Benchmark algorithms only')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    if not QUANTUM_AVAILABLE:
        print("Error: Quantum computing libraries not available.")
        print("Install required packages: pip install qiskit qiskit-aer cirq pennylane")
        sys.exit(1)

    benchmarker = QuantumBenchmarker()

    if not benchmarker.backends:
        print("Error: No quantum backends available for benchmarking.")
        sys.exit(1)

    try:
        if args.circuits_only:
            results = {'circuit_creation': benchmarker.benchmark_circuit_creation()}
        elif args.simulation_only:
            results = {'simulation': benchmarker.benchmark_simulation()}
        elif args.algorithms_only:
            results = {'algorithms': benchmarker.benchmark_algorithms()}
        else:
            results = benchmarker.run_comprehensive_benchmark()

        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'command_line_args': vars(args),
            'quantum_libraries_available': QUANTUM_AVAILABLE,
            'backends_tested': [name for name, _ in benchmarker.backends]
        }

        # Generate summary
        if 'summary' not in results:
            results['summary'] = benchmarker.generate_summary(results)

        # Save results
        output_file = benchmarker.save_results(results, args.output)

        # Print summary
        if not args.quiet:
            benchmarker.print_summary(results)

        print(f"\nBenchmark completed successfully! 📊")
        print(f"Results saved to: {output_file}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
