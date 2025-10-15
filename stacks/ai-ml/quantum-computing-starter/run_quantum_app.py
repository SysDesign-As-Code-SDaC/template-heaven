#!/usr/bin/env python3
"""
Quantum Computing Starter Application

Main entry point for running quantum computing experiments and demonstrations.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Computing Starter - Run quantum experiments and demonstrations"
    )

    parser.add_argument(
        "--demo",
        choices=["circuits", "shor", "qnn", "backends", "all"],
        help="Run specific demonstration"
    )

    parser.add_argument(
        "--backend",
        choices=["qiskit", "cirq", "pennylane"],
        default="qiskit",
        help="Quantum framework to use"
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots"
    )

    args = parser.parse_args()

    if args.demo == "circuits" or args.demo == "all":
        print("Running quantum circuits demonstration...")
        from src.quantum_circuits.basic_gates import demo_basic_circuits
        demo_basic_circuits()

    if args.demo == "shor" or args.demo == "all":
        print("\nRunning Shor's algorithm demonstration...")
        from src.quantum_circuits.algorithms.shor import demonstrate_shor
        demonstrate_shor()

    if args.demo == "qnn" or args.demo == "all":
        print("\nRunning quantum neural network demonstration...")
        from src.quantum_ml.qnn import demo_quantum_neural_network
        demo_quantum_neural_network()

    if args.demo == "backends" or args.demo == "all":
        print("\nRunning backend demonstration...")
        from src.backends.local_simulator import demo_simulators
        demo_simulators()

    if not args.demo:
        print("Quantum Computing Starter Template")
        print("===================================")
        print()
        print("Available demonstrations:")
        print("  --demo circuits    Basic quantum circuits (Bell states, GHZ, etc.)")
        print("  --demo shor        Shor's factoring algorithm")
        print("  --demo qnn         Quantum neural networks")
        print("  --demo backends    Quantum simulator backends")
        print("  --demo all         Run all demonstrations")
        print()
        print("Examples:")
        print("  python run_quantum_app.py --demo circuits")
        print("  python run_quantum_app.py --demo all --backend cirq")
        print()
        print("For interactive exploration, see the notebooks/ directory")


if __name__ == "__main__":
    main()
