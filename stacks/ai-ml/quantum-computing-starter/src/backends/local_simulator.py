"""
Local Quantum Simulator

This module provides a local quantum simulator backend for development and testing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

# Qiskit imports
try:
    from qiskit import Aer, QuantumCircuit, execute
    from qiskit.providers.basicaer import BasicAer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Cirq imports
try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

# PennyLane imports
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumBackend(ABC):
    """Abstract base class for quantum computing backends."""

    @abstractmethod
    def execute_circuit(self, circuit: Any, shots: int = 1024) -> Dict[str, Any]:
        """Execute a quantum circuit and return results."""
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the backend."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass


class QiskitSimulator(QuantumBackend):
    """Qiskit-based local quantum simulator."""

    def __init__(self, simulator_type: str = 'qasm'):
        """
        Initialize the Qiskit simulator.

        Args:
            simulator_type: Type of simulator ('qasm', 'statevector', 'basic')
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for QiskitSimulator")

        self.simulator_type = simulator_type

        if simulator_type == 'qasm':
            self.backend = Aer.get_backend('qasm_simulator')
        elif simulator_type == 'statevector':
            self.backend = Aer.get_backend('statevector_simulator')
        elif simulator_type == 'basic':
            self.backend = BasicAer.get_backend('qasm_simulator')
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")

    def execute_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Execute a quantum circuit.

        Args:
            circuit: Qiskit quantum circuit
            shots: Number of measurement shots

        Returns:
            Dictionary containing execution results
        """
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()

        if self.simulator_type == 'statevector':
            statevector = result.get_statevector(circuit)
            return {
                'statevector': statevector.data,
                'probabilities': np.abs(statevector.data) ** 2,
                'backend': 'qiskit_statevector',
                'shots': 1  # Statevector simulator doesn't use shots
            }
        else:
            counts = result.get_counts(circuit)
            return {
                'counts': dict(counts),
                'backend': f'qiskit_{self.simulator_type}',
                'shots': shots,
                'success': True
            }

    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            'name': f'Qiskit {self.simulator_type.title()} Simulator',
            'provider': 'Qiskit',
            'type': 'simulator',
            'max_shots': 8192,
            'max_qubits': 32,
            'available': True
        }

    def is_available(self) -> bool:
        """Check if backend is available."""
        return QISKIT_AVAILABLE


class CirqSimulator(QuantumBackend):
    """Cirq-based local quantum simulator."""

    def __init__(self):
        """Initialize the Cirq simulator."""
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq required for CirqSimulator")

        self.simulator = cirq.Simulator()

    def execute_circuit(self, circuit: cirq.Circuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Execute a quantum circuit.

        Args:
            circuit: Cirq quantum circuit
            shots: Number of measurement shots

        Returns:
            Dictionary containing execution results
        """
        result = self.simulator.run(circuit, repetitions=shots)

        # Convert measurements to counts format
        measurements = result.measurements
        counts = {}

        # Handle single measurement case
        if len(measurements) == 1:
            measurement_name = list(measurements.keys())[0]
            outcomes = measurements[measurement_name]

            for outcome in outcomes:
                bitstring = ''.join(str(int(bit)) for bit in outcome)
                counts[bitstring] = counts.get(bitstring, 0) + 1
        else:
            # Multiple measurements - flatten
            all_outcomes = []
            for measurement in measurements.values():
                all_outcomes.extend(measurement)

            for outcome in all_outcomes:
                bitstring = ''.join(str(int(bit)) for bit in outcome)
                counts[bitstring] = counts.get(bitstring, 0) + 1

        return {
            'counts': counts,
            'backend': 'cirq_simulator',
            'shots': shots,
            'success': True
        }

    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            'name': 'Cirq Simulator',
            'provider': 'Google',
            'type': 'simulator',
            'max_shots': 10000,
            'max_qubits': 26,
            'available': True
        }

    def is_available(self) -> bool:
        """Check if backend is available."""
        return CIRQ_AVAILABLE


class PennyLaneSimulator(QuantumBackend):
    """PennyLane-based local quantum simulator."""

    def __init__(self, n_qubits: int = 4):
        """
        Initialize the PennyLane simulator.

        Args:
            n_qubits: Number of qubits for the device
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane required for PennyLaneSimulator")

        self.n_qubits = n_qubits
        self.device = qml.device('default.qubit', wires=n_qubits)

    def execute_circuit(self, circuit: callable, shots: int = 1024) -> Dict[str, Any]:
        """
        Execute a quantum circuit (PennyLane QNode).

        Args:
            circuit: PennyLane QNode function
            shots: Number of measurement shots

        Returns:
            Dictionary containing execution results
        """
        # Execute the circuit
        result = circuit()

        # Convert result to standard format
        if isinstance(result, (int, float, complex)):
            # Single expectation value
            return {
                'expectation_value': float(result),
                'backend': 'pennylane_simulator',
                'shots': shots,
                'success': True
            }
        elif isinstance(result, (list, tuple, np.ndarray)):
            # Multiple expectation values or probabilities
            result_array = np.array(result)
            return {
                'expectation_values': result_array.tolist() if result_array.dtype in [np.complex64, np.complex128] else result_array.real.tolist(),
                'probabilities': (np.abs(result_array) ** 2).tolist() if result_array.dtype in [np.complex64, np.complex128] else None,
                'backend': 'pennylane_simulator',
                'shots': shots,
                'success': True
            }
        else:
            return {
                'result': str(result),
                'backend': 'pennylane_simulator',
                'shots': shots,
                'success': True
            }

    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information."""
        return {
            'name': 'PennyLane Simulator',
            'provider': 'Xanadu',
            'type': 'simulator',
            'max_shots': 10000,
            'max_qubits': self.n_qubits,
            'available': True
        }

    def is_available(self) -> bool:
        """Check if backend is available."""
        return PENNYLANE_AVAILABLE


def get_available_backends() -> List[QuantumBackend]:
    """
    Get all available local simulator backends.

    Returns:
        List of available backend instances
    """
    backends = []

    # Qiskit simulators
    if QISKIT_AVAILABLE:
        backends.extend([
            QiskitSimulator('qasm'),
            QiskitSimulator('statevector'),
        ])

    # Cirq simulator
    if CIRQ_AVAILABLE:
        backends.append(CirqSimulator())

    # PennyLane simulator
    if PENNYLANE_AVAILABLE:
        backends.append(PennyLaneSimulator())

    return backends


def benchmark_backends(circuit_creators: List[callable], shots: int = 1024) -> Dict[str, Any]:
    """
    Benchmark different backends with the same circuits.

    Args:
        circuit_creators: List of functions that create circuits for each backend
        shots: Number of measurement shots

    Returns:
        Benchmarking results
    """
    import time

    backends = get_available_backends()
    results = {}

    for i, backend in enumerate(backends):
        backend_name = backend.get_backend_info()['name']
        results[backend_name] = {}

        if i < len(circuit_creators):
            circuit_creator = circuit_creators[i]

            try:
                # Create circuit
                circuit = circuit_creator()

                # Time execution
                start_time = time.time()
                result = backend.execute_circuit(circuit, shots)
                execution_time = time.time() - start_time

                results[backend_name] = {
                    'execution_time': execution_time,
                    'success': result.get('success', False),
                    'error': None
                }

            except Exception as e:
                results[backend_name] = {
                    'execution_time': None,
                    'success': False,
                    'error': str(e)
                }

    return results


def demo_simulators():
    """Demonstrate different quantum simulators."""
    print("=== Quantum Simulator Demonstration ===\n")

    backends = get_available_backends()

    if not backends:
        print("No quantum simulators available!")
        print("Install Qiskit, Cirq, or PennyLane to use simulators.")
        return

    print(f"Available simulators: {len(backends)}")
    for backend in backends:
        info = backend.get_backend_info()
        print(f"- {info['name']} ({info['provider']})")

    print("\nTesting basic Bell state circuit...\n")

    # Test Qiskit backend
    if QISKIT_AVAILABLE:
        print("Qiskit Simulator:")
        try:
            from ..quantum_circuits.basic_gates import create_bell_state, measure_circuit

            bell_circuit = create_bell_state('qiskit')
            results = measure_circuit(bell_circuit, shots=1000, framework='qiskit')

            print(f"Measurement results: {results}")

            # Calculate Bell state fidelity
            expected_prob = 0.5
            fidelity = 0.0
            for outcome, count in results.items():
                if outcome in ['00', '11']:
                    fidelity += count / 1000

            print(".3f")

        except Exception as e:
            print(f"Qiskit test failed: {e}")

    # Test Cirq backend
    if CIRQ_AVAILABLE:
        print("\nCirq Simulator:")
        try:
            from ..quantum_circuits.basic_gates import create_bell_state

            bell_circuit = create_bell_state('cirq')
            backend = CirqSimulator()
            results = backend.execute_circuit(bell_circuit, shots=1000)

            print(f"Measurement results: {results['counts']}")

        except Exception as e:
            print(f"Cirq test failed: {e}")

    print("\nSimulator demonstration completed!")


if __name__ == "__main__":
    demo_simulators()
