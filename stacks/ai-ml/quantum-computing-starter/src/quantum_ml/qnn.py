"""
Quantum Neural Networks (QNN)

This module provides implementations of quantum neural networks
using PennyLane and Qiskit frameworks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# PennyLane imports
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

# Qiskit imports
try:
    from qiskit import QuantumCircuit
    from qiskit_machine_learning.neural_networks import CircuitQNN
    from qiskit_machine_learning.connectors import TorchConnector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class PennyLaneQNN(nn.Module):
    """
    Quantum Neural Network using PennyLane.

    This implements a hybrid classical-quantum neural network
    where classical layers feed into quantum circuits.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, n_classical_features: int = 10):
        """
        Initialize the quantum neural network.

        Args:
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of variational layers
            n_classical_features: Number of classical input features
        """
        super().__init__()

        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane required for PennyLaneQNN")

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical preprocessing layer
        self.classical_layer = nn.Linear(n_classical_features, n_qubits)

        # Quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)

        # Quantum circuit as a QNode
        @qml.qnode(self.dev, interface='torch')
        def quantum_circuit(inputs, weights):
            # Angle encoding of classical inputs
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

            # Variational quantum circuit
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            # Measure expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit

        # Initialize variational parameters
        weight_shapes = qml.templates.StronglyEntanglingLayers.shape(
            n_layers=n_layers, n_wires=n_qubits
        )
        self.quantum_weights = nn.Parameter(torch.randn(weight_shapes))

        # Output layer
        self.output_layer = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum neural network.

        Args:
            x: Input tensor of shape (batch_size, n_classical_features)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Classical preprocessing
        x = torch.relu(self.classical_layer(x))

        # Quantum processing
        quantum_output = self.quantum_circuit(x, self.quantum_weights)
        quantum_output = torch.stack(quantum_output, dim=1)

        # Classical postprocessing
        output = self.output_layer(quantum_output)
        return output


class QiskitQNN(nn.Module):
    """
    Quantum Neural Network using Qiskit.

    This implements a quantum neural network using Qiskit's
    CircuitQNN and connects it to PyTorch.
    """

    def __init__(self, n_qubits: int = 4, n_classical_features: int = 10):
        """
        Initialize the Qiskit quantum neural network.

        Args:
            n_qubits: Number of qubits
            n_classical_features: Number of classical input features
        """
        super().__init__()

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit Machine Learning required for QiskitQNN")

        self.n_qubits = n_qubits

        # Classical preprocessing
        self.classical_layer = nn.Linear(n_classical_features, n_qubits)

        # Create quantum circuit
        qc = QuantumCircuit(n_qubits)

        # Parameterized quantum circuit
        from qiskit.circuit import ParameterVector
        params = ParameterVector('theta', n_qubits * 3)  # 3 parameters per qubit

        # Add parameterized gates
        for i in range(n_qubits):
            qc.ry(params[i * 3], i)
            qc.rz(params[i * 3 + 1], i)
            qc.ry(params[i * 3 + 2], i)

        # Entangling gates
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        # Create CircuitQNN
        circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=params,
            weight_params=[],
            interpret=lambda x: np.arcsin(x),  # Map [-1, 1] to [-π/2, π/2]
            output_shape=1
        )

        # Connect to PyTorch
        self.quantum_layer = TorchConnector(circuit_qnn)

        # Output layer
        self.output_layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum neural network.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Classical preprocessing
        x = torch.relu(self.classical_layer(x))

        # Quantum processing (expects specific input format)
        # For simplicity, use the first feature
        quantum_input = x[:, 0:1]  # Take first feature
        quantum_output = self.quantum_layer(quantum_input)

        # Output layer
        output = self.output_layer(quantum_output)
        return output


def create_data_encoding_circuit(n_qubits: int, encoding_type: str = 'angle'):
    """
    Create a data encoding circuit for quantum machine learning.

    Args:
        n_qubits: Number of qubits
        encoding_type: Type of encoding ('angle', 'amplitude', 'basis')

    Returns:
        PennyLane QNode for data encoding
    """
    if not PENNYLANE_AVAILABLE:
        raise ImportError("PennyLane required for data encoding circuits")

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def encoding_circuit(data):
        """Encode classical data into quantum states."""

        if encoding_type == 'angle':
            # Angle encoding
            qml.templates.AngleEmbedding(data, wires=range(n_qubits))

        elif encoding_type == 'amplitude':
            # Amplitude encoding (for normalized vectors)
            qml.templates.AmplitudeEmbedding(data, wires=range(n_qubits), normalize=True)

        elif encoding_type == 'basis':
            # Basis encoding (for binary data)
            for i, bit in enumerate(data):
                if bit:
                    qml.PauliX(wires=i)

        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        return qml.state()

    return encoding_circuit


def train_quantum_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                       epochs: int = 100, lr: float = 0.01):
    """
    Train a quantum neural network model.

    Args:
        model: Quantum neural network model
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        lr: Learning rate
    """
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(".4f")

    print("Training completed!")


def demo_quantum_neural_network():
    """
    Demonstrate quantum neural network training and evaluation.
    """
    print("=== Quantum Neural Network Demonstration ===\n")

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)  # Linear relationship with noise

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train PennyLane QNN
    if PENNYLANE_AVAILABLE:
        print("Training PennyLane QNN...")
        try:
            qnn = PennyLaneQNN(n_qubits=4, n_layers=2, n_classical_features=10)
            train_quantum_model(qnn, X_train, y_train, epochs=50)

            # Evaluate
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                predictions = qnn(X_test_tensor).numpy().flatten()

            mse = np.mean((predictions - y_test) ** 2)
            print(".4f")

        except Exception as e:
            print(f"PennyLane QNN failed: {e}")

    # Train Qiskit QNN
    if QISKIT_AVAILABLE:
        print("\nTraining Qiskit QNN...")
        try:
            qnn_qiskit = QiskitQNN(n_qubits=4, n_classical_features=10)
            train_quantum_model(qnn_qiskit, X_train, y_train, epochs=30)

            # Evaluate
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                predictions = qnn_qiskit(X_test_tensor).numpy().flatten()

            mse = np.mean((predictions - y_test) ** 2)
            print(".4f")

        except Exception as e:
            print(f"Qiskit QNN failed: {e}")

    print("\nQuantum neural network demonstration completed!")


if __name__ == "__main__":
    demo_quantum_neural_network()
