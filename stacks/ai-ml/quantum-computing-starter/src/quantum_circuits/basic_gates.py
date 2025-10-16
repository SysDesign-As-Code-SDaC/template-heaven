"""
Basic Quantum Gates and Circuits

This module provides implementations of fundamental quantum gates and circuits
using Qiskit, Cirq, and PennyLane frameworks.
"""

import numpy as np
from typing import Optional, Union

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
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


def create_bell_state(framework: str = 'qiskit') -> Union[QuantumCircuit, cirq.Circuit, None]:
    """
    Create a Bell state (entangled two-qubit state).

    Args:
        framework: Quantum framework to use ('qiskit', 'cirq', 'pennylane')

    Returns:
        Bell state circuit in the specified framework

    Examples:
        >>> qc = create_bell_state('qiskit')
        >>> print(qc.draw())
    """
    if framework == 'qiskit' and QISKIT_AVAILABLE:
        qc = QuantumCircuit(2, 2)
        qc.h(0)  # Hadamard on first qubit
        qc.cx(0, 1)  # CNOT gate
        qc.measure_all()
        return qc

    elif framework == 'cirq' and CIRQ_AVAILABLE:
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(q0))
        circuit.append(cirq.CNOT(q0, q1))
        return circuit

    elif framework == 'pennylane' and PENNYLANE_AVAILABLE:
        @qml.qnode(qml.device('default.qubit', wires=2))
        def bell_state():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])
        return bell_state

    else:
        raise ValueError(f"Framework '{framework}' not available or not supported")


def create_ghz_state(n_qubits: int, framework: str = 'qiskit') -> Union[QuantumCircuit, cirq.Circuit, None]:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state.

    Args:
        n_qubits: Number of qubits in the GHZ state
        framework: Quantum framework to use

    Returns:
        GHZ state circuit
    """
    if framework == 'qiskit' and QISKIT_AVAILABLE:
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(0)  # Hadamard on first qubit
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)  # Chain of CNOT gates
        qc.measure_all()
        return qc

    elif framework == 'cirq' and CIRQ_AVAILABLE:
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        return circuit

    else:
        raise ValueError(f"Framework '{framework}' not available or not supported")


def create_quantum_fourier_transform(n_qubits: int, framework: str = 'qiskit') -> Union[QuantumCircuit, cirq.Circuit, None]:
    """
    Create a Quantum Fourier Transform circuit.

    Args:
        n_qubits: Number of qubits
        framework: Quantum framework to use

    Returns:
        QFT circuit
    """
    if framework == 'qiskit' and QISKIT_AVAILABLE:
        from qiskit.circuit.library import QFT
        qc = QFT(n_qubits)
        return qc

    elif framework == 'cirq' and CIRQ_AVAILABLE:
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()

        # QFT implementation
        for i in range(n_qubits):
            circuit.append(cirq.H(qubits[i]))
            for j in range(i + 1, n_qubits):
                angle = np.pi / (2 ** (j - i))
                circuit.append(cirq.CZ(qubits[j], qubits[i]) ** angle)

        return circuit

    else:
        raise ValueError(f"Framework '{framework}' not available or not supported")


def create_random_circuit(n_qubits: int, depth: int, framework: str = 'qiskit') -> Union[QuantumCircuit, cirq.Circuit, None]:
    """
    Create a random quantum circuit for supremacy experiments.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        framework: Quantum framework to use

    Returns:
        Random quantum circuit
    """
    if framework == 'qiskit' and QISKIT_AVAILABLE:
        qc = QuantumCircuit(n_qubits)
        np.random.seed(42)  # For reproducibility

        for layer in range(depth):
            # Random single-qubit rotations
            for qubit in range(n_qubits):
                if np.random.random() < 0.5:
                    qc.rx(np.random.random() * 2 * np.pi, qubit)
                if np.random.random() < 0.5:
                    qc.ry(np.random.random() * 2 * np.pi, qubit)
                if np.random.random() < 0.5:
                    qc.rz(np.random.random() * 2 * np.pi, qubit)

            # Random two-qubit gates
            for qubit in range(n_qubits - 1):
                if np.random.random() < 0.5:
                    qc.cx(qubit, qubit + 1)

        return qc

    elif framework == 'cirq' and CIRQ_AVAILABLE:
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        np.random.seed(42)

        for layer in range(depth):
            # Random single-qubit rotations
            for qubit in qubits:
                if np.random.random() < 0.5:
                    circuit.append(cirq.rx(np.random.random() * 2 * np.pi)(qubit))
                if np.random.random() < 0.5:
                    circuit.append(cirq.ry(np.random.random() * 2 * np.pi)(qubit))
                if np.random.random() < 0.5:
                    circuit.append(cirq.rz(np.random.random() * 2 * np.pi)(qubit))

            # Random two-qubit gates
            for i in range(n_qubits - 1):
                if np.random.random() < 0.5:
                    circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

        return circuit

    else:
        raise ValueError(f"Framework '{framework}' not available or not supported")


def measure_circuit(circuit: Union[QuantumCircuit, cirq.Circuit], shots: int = 1024, framework: str = 'qiskit') -> dict:
    """
    Execute a quantum circuit and return measurement results.

    Args:
        circuit: Quantum circuit to execute
        shots: Number of measurement shots
        framework: Framework being used

    Returns:
        Dictionary of measurement outcomes and their counts
    """
    if framework == 'qiskit' and QISKIT_AVAILABLE:
        from qiskit import Aer, execute

        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return dict(counts)

    elif framework == 'cirq' and CIRQ_AVAILABLE:
        # Simulate the circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=shots)

        # Convert to counts format
        measurements = result.measurements
        counts = {}
        for outcome in measurements.values():
            for measurement in outcome:
                bitstring = ''.join(str(bit) for bit in measurement)
                counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    else:
        raise ValueError(f"Framework '{framework}' not available or not supported")


# Example usage and testing functions
def demo_basic_circuits():
    """Demonstrate basic quantum circuits."""
    print("=== Quantum Computing Starter - Basic Circuits Demo ===\n")

    if QISKIT_AVAILABLE:
        print("1. Bell State (Qiskit):")
        bell_qc = create_bell_state('qiskit')
        print(bell_qc.draw(output='text'))

        print("\n2. Bell State Measurements:")
        counts = measure_circuit(bell_qc, shots=1000, framework='qiskit')
        print(counts)

    if CIRQ_AVAILABLE:
        print("\n3. GHZ State (Cirq):")
        ghz_circuit = create_ghz_state(3, 'cirq')
        print(ghz_circuit)

    if PENNYLANE_AVAILABLE:
        print("\n4. Bell State (PennyLane):")
        bell_pl = create_bell_state('pennylane')
        result = bell_pl()
        print(f"Bell state probabilities: {result}")


if __name__ == "__main__":
    demo_basic_circuits()
