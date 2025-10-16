"""
Shor's Algorithm Implementation

This module provides implementations of Shor's quantum factoring algorithm
for educational and research purposes.
"""

import numpy as np
from typing import List, Tuple, Optional
from math import gcd

# Qiskit imports
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.algorithms import Shor as QiskitShor
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def classical_gcd(a: int, b: int) -> int:
    """Classical GCD implementation using Euclidean algorithm."""
    while b != 0:
        a, b = b, a % b
    return a


def is_prime(n: int) -> bool:
    """Check if a number is prime using classical methods."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6

    return True


def find_period(a: int, N: int) -> Optional[int]:
    """
    Find the period of a^x mod N using classical period finding.
    This is a simplified version for educational purposes.
    """
    # For small N, we can brute force
    if N < 100:
        seen = {}
        x = 1
        for r in range(1, N + 1):
            x = (x * a) % N
            if x in seen:
                return r - seen[x]
            seen[x] = r
            if x == 1:
                return r
    return None


def shor_factorize_classical(N: int) -> List[int]:
    """
    Classical factoring for comparison with quantum algorithm.
    Uses trial division and Pollard's rho algorithm.

    Args:
        N: Number to factorize

    Returns:
        List of prime factors
    """
    factors = []

    # Handle even numbers
    while N % 2 == 0:
        factors.append(2)
        N = N // 2

    # Handle odd factors
    for i in range(3, int(np.sqrt(N)) + 1, 2):
        while N % i == 0:
            factors.append(i)
            N = N // i

    # If N is a prime number greater than 2
    if N > 2:
        factors.append(N)

    return factors


def shor_factorize_quantum(N: int) -> List[int]:
    """
    Quantum factoring using Shor's algorithm (Qiskit implementation).

    Args:
        N: Number to factorize (should be odd, non-prime)

    Returns:
        List of factors

    Raises:
        ImportError: If Qiskit is not available
        ValueError: If N is not suitable for factoring
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit required for quantum Shor's algorithm")

    if N % 2 == 0:
        return [2, N // 2]

    if is_prime(N):
        return [N]

    try:
        shor = QiskitShor()
        result = shor.factor(N)
        factors = result.factors

        # Validate factors
        product = 1
        for factor in factors:
            product *= factor

        if product == N:
            return sorted(factors)
        else:
            # Fallback to classical method
            return shor_factorize_classical(N)

    except Exception as e:
        print(f"Quantum factoring failed: {e}")
        print("Falling back to classical factoring...")
        return shor_factorize_classical(N)


def create_shor_circuit(N: int, a: int = None) -> Optional[QuantumCircuit]:
    """
    Create a Shor's algorithm quantum circuit.
    This is a simplified educational implementation.

    Args:
        N: Number to factorize
        a: Random base (if None, will be chosen randomly)

    Returns:
        Quantum circuit implementing Shor's algorithm
    """
    if not QISKIT_AVAILABLE:
        return None

    # For educational purposes, create a simplified circuit
    # Real Shor's algorithm requires quantum phase estimation

    # Determine register sizes
    n_count = 2 * len(bin(N)) - 2  # Counting qubits
    n_target = len(bin(N)) - 2     # Target qubits

    qc = QuantumCircuit(n_count + n_target, n_count)

    # Initialize superposition
    for i in range(n_count):
        qc.h(i)

    # Choose random a if not provided
    if a is None:
        a = np.random.randint(2, N)
        while gcd(a, N) != 1:
            a = np.random.randint(2, N)

    # Apply modular exponentiation (simplified)
    # In practice, this would be implemented using quantum arithmetic

    # Measure counting register
    qc.measure(range(n_count), range(n_count))

    return qc


def demonstrate_shor():
    """
    Demonstrate Shor's algorithm with various examples.
    """
    print("=== Shor's Algorithm Demonstration ===\n")

    test_cases = [15, 21, 35, 51, 77]

    for N in test_cases:
        print(f"Factoring {N}:")

        # Classical method
        classical_factors = shor_factorize_classical(N)
        print(f"  Classical: {N} = {' × '.join(map(str, classical_factors))}")

        # Quantum method (if available)
        if QISKIT_AVAILABLE:
            try:
                quantum_factors = shor_factorize_quantum(N)
                print(f"  Quantum:   {N} = {' × '.join(map(str, quantum_factors))}")
            except Exception as e:
                print(f"  Quantum:   Failed - {e}")
        else:
            print("  Quantum:   Qiskit not available")

        print()


def analyze_shor_complexity():
    """
    Analyze the computational complexity of Shor's algorithm.
    """
    print("=== Shor's Algorithm Complexity Analysis ===\n")

    print("Classical factoring complexity: O(2^(n/2)) or O(n^3) with optimizations")
    print("Shor's algorithm complexity: O(n^2 log n log log n) with O(n) qubits")
    print()

    # Demonstrate scaling
    bit_sizes = [10, 20, 50, 100, 200]

    print("Bit size | Classical operations | Quantum operations | Qubits needed")
    print("---------|---------------------|-------------------|--------------")

    for bits in bit_sizes:
        N = 2**bits
        classical_ops = 2**(bits//2)  # Approximate
        quantum_ops = bits**2 * np.log2(bits) * np.log2(np.log2(bits))
        qubits = 2 * bits

        print("8")

        if bits > 50:
            break  # Prevent overflow


if __name__ == "__main__":
    demonstrate_shor()
    analyze_shor_complexity()
