"""
Basic tests for quantum computing starter template.
"""

import pytest
import numpy as np

# Test imports
try:
    from src.quantum_circuits.basic_gates import create_bell_state, create_ghz_state
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class TestBasicCircuits:
    """Test basic quantum circuit functionality."""

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
    def test_bell_state_qiskit(self):
        """Test Bell state creation with Qiskit."""
        from qiskit import QuantumCircuit

        bell_circuit = create_bell_state('qiskit')
        assert isinstance(bell_circuit, QuantumCircuit)
        assert bell_circuit.num_qubits == 2
        assert bell_circuit.num_clbits == 2

    @pytest.mark.skipif(not CIRQ_AVAILABLE, reason="Cirq not available")
    def test_bell_state_cirq(self):
        """Test Bell state creation with Cirq."""

        bell_circuit = create_bell_state('cirq')
        assert isinstance(bell_circuit, cirq.Circuit)

        # Check that circuit has expected gates
        operations = list(bell_circuit.all_operations())
        assert len(operations) >= 2  # H + CNOT

    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available")
    def test_bell_state_pennylane(self):
        """Test Bell state creation with PennyLane."""

        bell_circuit = create_bell_state('pennylane')
        assert callable(bell_circuit)

        # Test execution
        result = bell_circuit()
        assert len(result) == 4  # 2^2 probabilities for 2 qubits

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
    def test_ghz_state(self):
        """Test GHZ state creation."""
        from qiskit import QuantumCircuit

        ghz_circuit = create_ghz_state(3, 'qiskit')
        assert isinstance(ghz_circuit, QuantumCircuit)
        assert ghz_circuit.num_qubits == 3

    def test_invalid_framework(self):
        """Test error handling for invalid framework."""
        with pytest.raises(ValueError, match="not available or not supported"):
            create_bell_state('invalid_framework')


class TestShorAlgorithm:
    """Test Shor's algorithm implementation."""

    def test_classical_gcd(self):
        """Test classical GCD function."""
        from src.quantum_circuits.algorithms.shor import classical_gcd

        assert classical_gcd(48, 18) == 6
        assert classical_gcd(100, 75) == 25
        assert classical_gcd(7, 11) == 1

    def test_is_prime(self):
        """Test prime checking function."""
        from src.quantum_circuits.algorithms.shor import is_prime

        assert is_prime(2) == True
        assert is_prime(3) == True
        assert is_prime(4) == False
        assert is_prime(17) == True
        assert is_prime(18) == False

    def test_classical_factorization(self):
        """Test classical factorization."""
        from src.quantum_circuits.algorithms.shor import shor_factorize_classical

        # Test various numbers
        assert sorted(shor_factorize_classical(15)) == [3, 5]
        assert sorted(shor_factorize_classical(21)) == [3, 7]
        assert sorted(shor_factorize_classical(35)) == [5, 7]


class TestBackends:
    """Test quantum backend functionality."""

    @pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
    def test_qiskit_simulator(self):
        """Test Qiskit simulator backend."""
        from src.backends.local_simulator import QiskitSimulator
        from src.quantum_circuits.basic_gates import create_bell_state

        backend = QiskitSimulator('qasm')
        assert backend.is_available()

        bell_circuit = create_bell_state('qiskit')
        result = backend.execute_circuit(bell_circuit, shots=100)

        assert 'counts' in result
        assert 'backend' in result
        assert result['shots'] == 100
        assert result['success'] == True

    @pytest.mark.skipif(not CIRQ_AVAILABLE, reason="Cirq not available")
    def test_cirq_simulator(self):
        """Test Cirq simulator backend."""
        from src.backends.local_simulator import CirqSimulator
        from src.quantum_circuits.basic_gates import create_bell_state

        backend = CirqSimulator()
        assert backend.is_available()

        bell_circuit = create_bell_state('cirq')
        result = backend.execute_circuit(bell_circuit, shots=100)

        assert 'counts' in result
        assert 'backend' in result
        assert result['shots'] == 100
        assert result['success'] == True


class TestQuantumML:
    """Test quantum machine learning components."""

    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available")
    def test_pennylane_qnn_initialization(self):
        """Test PennyLane QNN initialization."""
        from src.quantum_ml.qnn import PennyLaneQNN

        qnn = PennyLaneQNN(n_qubits=4, n_layers=2, n_classical_features=10)
        assert qnn.n_qubits == 4
        assert qnn.n_layers == 2

    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available")
    def test_data_encoding_circuit(self):
        """Test data encoding circuit creation."""
        from src.quantum_ml.qnn import create_data_encoding_circuit

        encoding_circuit = create_data_encoding_circuit(4, 'angle')
        assert callable(encoding_circuit)

        # Test with sample data
        data = np.array([0.1, 0.2, 0.3, 0.4])
        result = encoding_circuit(data)
        assert len(result) == 2**4  # 2^4 = 16 for 4 qubits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
