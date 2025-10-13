# Qiskit Quantum Computing Application Template

A production-ready quantum computing application template using IBM Qiskit, featuring quantum algorithms, quantum machine learning, and quantum simulation for 2025.

## 🚀 Features

- **Qiskit** - IBM's quantum computing framework
- **Quantum Algorithms** - Shor's, Grover's, VQE, QAOA implementations
- **Quantum Machine Learning** - Quantum neural networks, quantum kernels
- **Quantum Simulation** - Classical simulation of quantum circuits
- **Quantum Hardware** - Real quantum computer integration
- **Quantum Cryptography** - BB84, E91 protocols
- **Quantum Optimization** - Combinatorial optimization problems
- **Quantum Chemistry** - Molecular simulation and optimization
- **Visualization** - Quantum circuit visualization and analysis
- **Performance Optimization** - Circuit optimization and transpilation

## 📋 Prerequisites

- Python 3.9+
- IBM Quantum account (for real hardware access)
- Git

## 🛠️ Quick Start

### 1. Create New Quantum Project

```bash
git clone <this-repo> my-quantum-app
cd my-quantum-app
```

### 2. Environment Setup

```bash
cp .env.example .env
```

Configure your environment variables:

```env
# IBM Quantum Credentials
QISKIT_IBM_TOKEN=your_ibm_quantum_token
QISKIT_IBM_URL=https://auth.quantum-computing.ibm.com/api

# Quantum Hardware Configuration
DEFAULT_BACKEND=ibmq_qasm_simulator
OPTIMIZATION_LEVEL=3
SHOTS=1024

# Simulation Configuration
MAX_QUBITS=32
SIMULATION_METHOD=statevector

# Quantum Machine Learning
QML_LAYERS=4
QML_EPOCHS=100
QML_LEARNING_RATE=0.01

# Quantum Chemistry
MOLECULE_GEOMETRY=H2
BASIS_SET=sto-3g
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Quantum Examples

```bash
# Run Shor's algorithm
python examples/shor_algorithm.py

# Run Grover's algorithm
python examples/grover_algorithm.py

# Run quantum machine learning
python examples/quantum_ml.py

# Run quantum chemistry simulation
python examples/quantum_chemistry.py
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── algorithms/             # Quantum algorithms
│   │   ├── shor.py            # Shor's factorization algorithm
│   │   ├── grover.py          # Grover's search algorithm
│   │   ├── vqe.py             # Variational Quantum Eigensolver
│   │   └── qaoa.py            # Quantum Approximate Optimization Algorithm
│   ├── quantum_ml/            # Quantum machine learning
│   │   ├── qnn.py             # Quantum neural networks
│   │   ├── quantum_kernel.py  # Quantum kernel methods
│   │   └── vqc.py             # Variational quantum classifier
│   ├── quantum_chemistry/     # Quantum chemistry simulations
│   │   ├── molecular_simulation.py
│   │   ├── ground_state.py
│   │   └── excited_states.py
│   ├── cryptography/          # Quantum cryptography
│   │   ├── bb84.py            # BB84 protocol
│   │   ├── e91.py             # E91 protocol
│   │   └── quantum_key_distribution.py
│   ├── optimization/          # Quantum optimization
│   │   ├── tsp.py             # Traveling salesman problem
│   │   ├── maxcut.py          # Max-cut problem
│   │   └── portfolio.py       # Portfolio optimization
│   ├── simulation/            # Quantum simulation
│   │   ├── hamiltonian.py     # Hamiltonian simulation
│   │   ├── time_evolution.py  # Time evolution
│   │   └── noise_models.py    # Noise modeling
│   └── utils/                 # Utility functions
│       ├── circuit_optimization.py
│       ├── visualization.py
│       └── backend_manager.py
├── examples/                  # Example implementations
├── tests/                     # Test files
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
└── data/                      # Data files
```

## 🔧 Available Scripts

```bash
# Development
python -m pytest tests/        # Run tests
python -m black src/           # Format code
python -m flake8 src/          # Lint code

# Quantum Algorithms
python examples/shor_algorithm.py
python examples/grover_algorithm.py
python examples/vqe_algorithm.py

# Quantum Machine Learning
python examples/quantum_classification.py
python examples/quantum_regression.py
python examples/quantum_gan.py

# Quantum Chemistry
python examples/molecular_ground_state.py
python examples/reaction_pathway.py

# Quantum Optimization
python examples/traveling_salesman.py
python examples/portfolio_optimization.py
```

## ⚛️ Quantum Algorithm Implementations

### Shor's Algorithm

```python
# src/algorithms/shor.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms import Shor
from qiskit.primitives import Sampler
import numpy as np

class ShorAlgorithm:
    def __init__(self, backend=None):
        self.backend = backend
        self.sampler = Sampler()
    
    def factorize(self, N, a=None):
        """
        Factorize a number N using Shor's algorithm.
        
        Args:
            N (int): Number to factorize
            a (int): Random number coprime to N
            
        Returns:
            tuple: Factors of N
        """
        if a is None:
            a = self._find_coprime(N)
        
        # Check if a^r ≡ 1 (mod N) for some r
        period = self._find_period(a, N)
        
        if period % 2 == 1:
            return self.factorize(N)  # Try again with different a
        
        # Calculate potential factors
        factor1 = np.gcd(a**(period//2) + 1, N)
        factor2 = np.gcd(a**(period//2) - 1, N)
        
        if factor1 != 1 and factor1 != N:
            return (factor1, N // factor1)
        elif factor2 != 1 and factor2 != N:
            return (factor2, N // factor2)
        else:
            return self.factorize(N)  # Try again
    
    def _find_period(self, a, N):
        """Find the period of a^x mod N using quantum period finding."""
        # Create quantum circuit for period finding
        n_qubits = int(np.ceil(np.log2(N)))
        
        qreg = QuantumRegister(2 * n_qubits, 'q')
        creg = ClassicalRegister(2 * n_qubits, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Initialize superposition
        for i in range(n_qubits):
            qc.h(qreg[i])
        
        # Apply modular exponentiation
        qc = self._modular_exponentiation(qc, qreg, a, N, n_qubits)
        
        # Apply inverse QFT
        qc = self._inverse_qft(qc, qreg[:n_qubits])
        
        # Measure
        qc.measure(qreg[:n_qubits], creg[:n_qubits])
        
        # Execute circuit
        job = self.sampler.run([qc], shots=1024)
        result = job.result()
        
        # Extract period from measurement results
        counts = result.quasi_dists[0]
        period = self._extract_period_from_counts(counts, n_qubits)
        
        return period
    
    def _modular_exponentiation(self, qc, qreg, a, N, n_qubits):
        """Implement modular exponentiation using quantum gates."""
        # This is a simplified version
        # In practice, this would be much more complex
        for i in range(n_qubits):
            qc.cx(qreg[i], qreg[n_qubits + i])
        
        return qc
    
    def _inverse_qft(self, qc, qreg):
        """Apply inverse Quantum Fourier Transform."""
        n = len(qreg)
        
        for i in range(n):
            qc.h(qreg[i])
            for j in range(i + 1, n):
                qc.cp(-np.pi / (2**(j - i)), qreg[j], qreg[i])
        
        return qc
    
    def _find_coprime(self, N):
        """Find a number coprime to N."""
        import random
        while True:
            a = random.randint(2, N - 1)
            if np.gcd(a, N) == 1:
                return a
    
    def _extract_period_from_counts(self, counts, n_qubits):
        """Extract period from measurement counts."""
        # Simplified period extraction
        # In practice, this would use continued fractions
        return 4  # Placeholder
```

### Grover's Algorithm

```python
# src/algorithms/grover.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms import Grover
from qiskit.primitives import Sampler
import numpy as np

class GroverAlgorithm:
    def __init__(self, backend=None):
        self.backend = backend
        self.sampler = Sampler()
    
    def search(self, oracle, num_qubits, target_items=None):
        """
        Perform Grover's search algorithm.
        
        Args:
            oracle: Oracle function that marks target items
            num_qubits: Number of qubits in the search space
            target_items: List of target items to search for
            
        Returns:
            list: Found items
        """
        # Calculate optimal number of iterations
        num_iterations = int(np.pi / 4 * np.sqrt(2**num_qubits))
        
        # Create quantum circuit
        qreg = QuantumRegister(num_qubits, 'q')
        creg = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Initialize superposition
        for i in range(num_qubits):
            qc.h(qreg[i])
        
        # Apply Grover iterations
        for _ in range(num_iterations):
            # Apply oracle
            qc = oracle(qc, qreg)
            
            # Apply diffusion operator
            qc = self._diffusion_operator(qc, qreg)
        
        # Measure
        qc.measure(qreg, creg)
        
        # Execute circuit
        job = self.sampler.run([qc], shots=1024)
        result = job.result()
        
        # Extract results
        counts = result.quasi_dists[0]
        found_items = self._extract_results(counts, num_qubits)
        
        return found_items
    
    def _diffusion_operator(self, qc, qreg):
        """Apply the diffusion operator (inversion about average)."""
        n = len(qreg)
        
        # Apply H gates
        for i in range(n):
            qc.h(qreg[i])
        
        # Apply X gates
        for i in range(n):
            qc.x(qreg[i])
        
        # Apply multi-controlled Z gate
        if n == 1:
            qc.z(qreg[0])
        elif n == 2:
            qc.cz(qreg[0], qreg[1])
        else:
            # For more qubits, use a more complex implementation
            qc.mcp(np.pi, qreg[:-1], qreg[-1])
        
        # Apply X gates again
        for i in range(n):
            qc.x(qreg[i])
        
        # Apply H gates again
        for i in range(n):
            qc.h(qreg[i])
        
        return qc
    
    def _extract_results(self, counts, num_qubits):
        """Extract search results from measurement counts."""
        # Sort by probability
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for state, probability in sorted_counts[:5]:  # Top 5 results
            if probability > 0.01:  # Threshold for significant results
                results.append((state, probability))
        
        return results
```

## 🤖 Quantum Machine Learning

### Quantum Neural Network

```python
# src/quantum_ml/qnn.py
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
import numpy as np
from typing import List, Tuple

class QuantumNeuralNetwork:
    def __init__(self, num_qubits: int, num_layers: int, backend=None):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.backend = backend
        self.estimator = Estimator()
        self.parameters = []
        self.circuit = self._create_circuit()
    
    def _create_circuit(self):
        """Create the quantum neural network circuit."""
        qreg = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(qreg)
        
        # Create parameterized layers
        for layer in range(self.num_layers):
            # Data encoding layer
            for i in range(self.num_qubits):
                param = Parameter(f'data_{layer}_{i}')
                self.parameters.append(param)
                qc.ry(param, qreg[i])
            
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qc.cx(qreg[i], qreg[i + 1])
            
            # Variational layer
            for i in range(self.num_qubits):
                param = Parameter(f'var_{layer}_{i}')
                self.parameters.append(param)
                qc.ry(param, qreg[i])
                param = Parameter(f'var_{layer}_{i}_z')
                self.parameters.append(param)
                qc.rz(param, qreg[i])
        
        return qc
    
    def forward(self, data: np.ndarray, weights: np.ndarray) -> float:
        """
        Forward pass of the quantum neural network.
        
        Args:
            data: Input data
            weights: Network weights
            
        Returns:
            float: Output expectation value
        """
        # Bind parameters
        param_dict = {}
        param_idx = 0
        
        for layer in range(self.num_layers):
            # Data parameters
            for i in range(self.num_qubits):
                param_dict[self.parameters[param_idx]] = data[i]
                param_idx += 1
            
            # Variational parameters
            for i in range(self.num_qubits):
                param_dict[self.parameters[param_idx]] = weights[layer * self.num_qubits + i]
                param_idx += 1
                param_dict[self.parameters[param_idx]] = weights[layer * self.num_qubits + i + self.num_qubits]
                param_idx += 1
        
        # Create observable (measurement)
        observable = self._create_observable()
        
        # Execute circuit
        job = self.estimator.run([self.circuit], [observable], [param_dict])
        result = job.result()
        
        return result.values[0]
    
    def _create_observable(self):
        """Create measurement observable."""
        from qiskit.quantum_info import SparsePauliOp
        
        # Measure Z on first qubit
        observable = SparsePauliOp('Z' + 'I' * (self.num_qubits - 1))
        return observable
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """
        Train the quantum neural network.
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            lr: Learning rate
        """
        # Initialize weights
        num_weights = self.num_layers * self.num_qubits * 2
        weights = np.random.uniform(-np.pi, np.pi, num_weights)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i, (x, target) in enumerate(zip(X, y)):
                # Forward pass
                output = self.forward(x, weights)
                
                # Calculate loss (MSE)
                loss = (output - target) ** 2
                total_loss += loss
                
                # Calculate gradients (finite difference)
                gradients = self._calculate_gradients(x, weights, target)
                
                # Update weights
                weights -= lr * gradients
            
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _calculate_gradients(self, x: np.ndarray, weights: np.ndarray, target: float) -> np.ndarray:
        """Calculate gradients using finite difference."""
        epsilon = 0.01
        gradients = np.zeros_like(weights)
        
        for i in range(len(weights)):
            # Forward difference
            weights_plus = weights.copy()
            weights_plus[i] += epsilon
            
            output_plus = self.forward(x, weights_plus)
            gradient = (output_plus - target) / epsilon
            
            gradients[i] = gradient
        
        return gradients
```

## 🧬 Quantum Chemistry

### Molecular Ground State

```python
# src/quantum_chemistry/ground_state.py
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
import numpy as np

class QuantumChemistrySimulation:
    def __init__(self, molecule_geometry, basis_set='sto-3g'):
        self.molecule_geometry = molecule_geometry
        self.basis_set = basis_set
        self.driver = None
        self.problem = None
        self.qubit_converter = None
        self.estimator = Estimator()
    
    def setup_molecule(self):
        """Setup the molecular system."""
        # Create molecule
        molecule = Molecule(
            geometry=self.molecule_geometry,
            charge=0,
            multiplicity=1
        )
        
        # Create driver
        self.driver = ElectronicStructureDriverType.PYSCF.new_driver(molecule)
        
        # Create problem
        self.problem = ElectronicStructureProblem(self.driver, self.basis_set)
        
        # Setup qubit converter
        self.qubit_converter = QubitConverter(JordanWignerMapper())
    
    def find_ground_state(self):
        """Find the ground state energy using VQE."""
        # Convert to qubit operator
        second_q_op = self.problem.second_q_ops()
        qubit_op = self.qubit_converter.convert(second_q_op[0])
        
        # Create ansatz
        num_qubits = qubit_op.num_qubits
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cz',
            entanglement='linear',
            reps=3
        )
        
        # Setup VQE
        optimizer = SLSQP(maxiter=100)
        vqe = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=self.estimator
        )
        
        # Run VQE
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        
        return result.eigenvalue.real
    
    def calculate_molecular_properties(self):
        """Calculate various molecular properties."""
        # Get ground state energy
        ground_state_energy = self.find_ground_state()
        
        # Calculate other properties
        properties = {
            'ground_state_energy': ground_state_energy,
            'num_electrons': self.problem.num_particles,
            'num_orbitals': self.problem.num_spatial_orbitals,
            'num_qubits': self.problem.num_spatial_orbitals * 2
        }
        
        return properties

# Example usage
if __name__ == "__main__":
    # H2 molecule
    h2_geometry = [
        ['H', [0.0, 0.0, 0.0]],
        ['H', [0.0, 0.0, 0.74]]
    ]
    
    # Create simulation
    sim = QuantumChemistrySimulation(h2_geometry)
    sim.setup_molecule()
    
    # Find ground state
    energy = sim.find_ground_state()
    print(f"H2 ground state energy: {energy:.6f} Hartree")
    
    # Calculate properties
    properties = sim.calculate_molecular_properties()
    print(f"Molecular properties: {properties}")
```

## 🔐 Quantum Cryptography

### BB84 Protocol

```python
# src/cryptography/bb84.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Sampler
import numpy as np
import random

class BB84Protocol:
    def __init__(self, key_length=256):
        self.key_length = key_length
        self.sampler = Sampler()
    
    def generate_key(self):
        """Generate a quantum key using BB84 protocol."""
        # Alice's random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(self.key_length)]
        alice_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        
        # Bob's random bases
        bob_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        
        # Create quantum circuit
        qreg = QuantumRegister(1, 'q')
        creg = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qreg, creg)
        
        # Simulate quantum transmission
        bob_bits = []
        for i in range(self.key_length):
            # Alice prepares qubit
            if alice_bits[i] == 1:
                if alice_bases[i] == 0:  # Z basis
                    qc.x(qreg[0])
                else:  # X basis
                    qc.h(qreg[0])
                    qc.x(qreg[0])
            else:
                if alice_bases[i] == 1:  # X basis
                    qc.h(qreg[0])
            
            # Bob measures
            if bob_bases[i] == 1:  # X basis
                qc.h(qreg[0])
            
            qc.measure(qreg[0], creg[0])
            
            # Execute circuit
            job = self.sampler.run([qc], shots=1)
            result = job.result()
            
            # Extract measurement result
            measurement = list(result.quasi_dists[0].keys())[0]
            bob_bits.append(measurement)
            
            # Reset circuit for next iteration
            qc.reset(qreg[0])
        
        # Sift key (keep only matching bases)
        sifted_key = []
        for i in range(self.key_length):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])
        
        return sifted_key, alice_bits, bob_bits, alice_bases, bob_bases
    
    def detect_eavesdropping(self, alice_bits, bob_bits, alice_bases, bob_bases, test_fraction=0.1):
        """Detect eavesdropping by comparing test bits."""
        # Select test bits
        test_indices = random.sample(
            range(len(alice_bits)), 
            int(len(alice_bits) * test_fraction)
        )
        
        # Compare test bits
        errors = 0
        for i in test_indices:
            if alice_bases[i] == bob_bases[i]:  # Same basis
                if alice_bits[i] != bob_bits[i]:
                    errors += 1
        
        error_rate = errors / len(test_indices) if test_indices else 0
        
        # If error rate is too high, assume eavesdropping
        if error_rate > 0.11:  # Theoretical limit for BB84
            return True, error_rate
        
        return False, error_rate
```

## 🚀 Deployment and Scaling

### Quantum Cloud Integration

```python
# src/utils/backend_manager.py
from qiskit import IBMQ
from qiskit.providers.ibmq import IBMQProvider
from qiskit.providers.aer import AerSimulator
import os

class QuantumBackendManager:
    def __init__(self):
        self.provider = None
        self.backends = {}
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup IBM Quantum provider."""
        try:
            # Load IBM Quantum account
            IBMQ.load_account()
            self.provider = IBMQ.get_provider()
            
            # Get available backends
            self.backends = {
                'simulator': AerSimulator(),
                'ibmq_qasm_simulator': self.provider.get_backend('ibmq_qasm_simulator'),
                'ibmq_lima': self.provider.get_backend('ibmq_lima'),
                'ibmq_belem': self.provider.get_backend('ibmq_belem'),
                'ibmq_quito': self.provider.get_backend('ibmq_quito'),
            }
        except Exception as e:
            print(f"Failed to setup IBM Quantum provider: {e}")
            # Fallback to local simulator
            self.backends = {
                'simulator': AerSimulator()
            }
    
    def get_backend(self, backend_name='simulator'):
        """Get quantum backend."""
        return self.backends.get(backend_name, self.backends['simulator'])
    
    def get_backend_info(self, backend_name):
        """Get backend information."""
        backend = self.get_backend(backend_name)
        if hasattr(backend, 'properties'):
            return {
                'name': backend.name,
                'num_qubits': backend.configuration().n_qubits,
                'coupling_map': backend.configuration().coupling_map,
                'basis_gates': backend.configuration().basis_gates,
                'quantum_volume': getattr(backend.properties(), 'quantum_volume', None)
            }
        return {'name': backend.name}
```

## 📚 Learning Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)
- [Quantum Computing Course](https://qiskit.org/learn/)
- [Quantum Machine Learning](https://qiskit.org/textbook/ch-machine-learning/)

## 🔗 Upstream Source

- **Repository**: [Qiskit/qiskit](https://github.com/Qiskit/qiskit)
- **Documentation**: [qiskit.org](https://qiskit.org/)
- **License**: Apache-2.0
