#!/usr/bin/env python3
"""
Quantum Computing Environment Setup Script

This script automates the setup of a complete quantum computing development environment,
including dependency installation, environment configuration, and validation.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from backends.local_simulator import get_available_backends
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


class QuantumEnvironmentSetup:
    """Automated quantum computing environment setup."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        self.installed_packages = []
        self.failed_packages = []

    def log(self, message: str, level: str = "INFO"):
        """Log messages with optional verbosity."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅ ",
                "WARNING": "⚠️ ",
                "ERROR": "❌ "
            }.get(level, "")
            print(f"{prefix}{message}")

    def run_command(self, command: List[str], cwd: Optional[Path] = None,
                   capture_output: bool = False) -> Tuple[int, str, str]:
        """
        Run shell command with proper error handling.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            self.log(f"Running: {' '.join(command)}", "INFO")

            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=capture_output,
                text=True,
                check=False
            )

            return result.returncode, result.stdout, result.stderr

        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            return -1, "", str(e)

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        self.log("Checking Python version...")

        version = sys.version_info
        required_version = (3, 9)

        if version >= required_version:
            self.log(f"Python {version.major}.{version.minor}.{version.micro} ✓", "SUCCESS")
            return True
        else:
            self.log(f"Python {version.major}.{version.minor}.{version.micro} is too old. "
                    f"Required: {required_version[0]}.{required_version[1]}+", "ERROR")
            return False

    def install_package(self, package_name: str, description: str = "") -> bool:
        """Install a Python package via pip."""
        desc = f" ({description})" if description else ""
        self.log(f"Installing {package_name}{desc}...")

        returncode, stdout, stderr = self.run_command(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True
        )

        if returncode == 0:
            self.installed_packages.append(package_name)
            self.log(f"Successfully installed {package_name}", "SUCCESS")
            return True
        else:
            self.failed_packages.append(package_name)
            self.log(f"Failed to install {package_name}: {stderr}", "ERROR")
            return False

    def install_quantum_frameworks(self) -> bool:
        """Install quantum computing frameworks."""
        self.log("Installing quantum computing frameworks...")

        frameworks = [
            ("qiskit", "IBM Quantum framework"),
            ("qiskit-aer", "Qiskit simulator backend"),
            ("qiskit-ibmq-provider", "IBM Quantum cloud access"),
            ("qiskit-machine-learning", "Quantum machine learning"),
            ("cirq", "Google Cirq framework"),
            ("cirq-google", "Google Quantum AI integration"),
            ("pennylane", "Xanadu PennyLane framework"),
            ("pennylane-qiskit", "PennyLane-Qiskit integration")
        ]

        success_count = 0
        for package, description in frameworks:
            if self.install_package(package, description):
                success_count += 1

        self.log(f"Installed {success_count}/{len(frameworks)} quantum frameworks", "SUCCESS")
        return success_count > 0

    def install_development_tools(self) -> bool:
        """Install development and testing tools."""
        self.log("Installing development tools...")

        dev_packages = [
            ("pytest", "Testing framework"),
            ("pytest-asyncio", "Async testing support"),
            ("pytest-cov", "Coverage reporting"),
            ("black", "Code formatting"),
            ("isort", "Import sorting"),
            ("flake8", "Linting"),
            ("mypy", "Type checking"),
            ("jupyter", "Interactive notebooks"),
            ("ipykernel", "Jupyter kernel"),
            ("matplotlib", "Plotting library"),
            ("numpy", "Numerical computing"),
            ("scipy", "Scientific computing")
        ]

        success_count = 0
        for package, description in dev_packages:
            if self.install_package(package, description):
                success_count += 1

        self.log(f"Installed {success_count}/{len(dev_packages)} development tools", "SUCCESS")
        return success_count >= len(dev_packages) * 0.8  # 80% success rate

    def validate_installation(self) -> Dict[str, bool]:
        """Validate that quantum frameworks are working."""
        self.log("Validating quantum framework installations...")

        validation_results = {}

        # Test Qiskit
        try:
            import qiskit
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            validation_results['qiskit'] = True
            self.log("Qiskit validation successful", "SUCCESS")
        except ImportError:
            validation_results['qiskit'] = False
            self.log("Qiskit validation failed", "ERROR")

        # Test Cirq
        try:
            import cirq
            q0, q1 = cirq.LineQubit.range(2)
            circuit = cirq.Circuit()
            circuit.append(cirq.H(q0))
            circuit.append(cirq.CNOT(q0, q1))
            validation_results['cirq'] = True
            self.log("Cirq validation successful", "SUCCESS")
        except ImportError:
            validation_results['cirq'] = False
            self.log("Cirq validation failed", "ERROR")

        # Test PennyLane
        try:
            import pennylane as qml
            dev = qml.device('default.qubit', wires=2)
            @qml.qnode(dev)
            def test_circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=[0, 1])
            result = test_circuit()
            validation_results['pennylane'] = len(result) == 4
            self.log("PennyLane validation successful", "SUCCESS")
        except (ImportError, Exception):
            validation_results['pennylane'] = False
            self.log("PennyLane validation failed", "ERROR")

        # Test project imports
        try:
            from quantum_circuits.basic_gates import create_bell_state
            bell = create_bell_state('qiskit')
            validation_results['project_imports'] = True
            self.log("Project imports validation successful", "SUCCESS")
        except ImportError as e:
            validation_results['project_imports'] = False
            self.log(f"Project imports validation failed: {e}", "ERROR")

        return validation_results

    def setup_jupyter_kernel(self) -> bool:
        """Setup Jupyter kernel for quantum computing."""
        self.log("Setting up Jupyter kernel...")

        try:
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "ipykernel", "install",
                "--user", "--name", "quantum-computing", "--display-name", "Quantum Computing"
            ], capture_output=True)

            if returncode == 0:
                self.log("Jupyter kernel setup successful", "SUCCESS")
                return True
            else:
                self.log(f"Jupyter kernel setup failed: {stderr}", "ERROR")
                return False

        except Exception as e:
            self.log(f"Jupyter kernel setup error: {e}", "ERROR")
            return False

    def create_env_file(self) -> bool:
        """Create environment configuration file."""
        self.log("Creating environment configuration...")

        env_path = self.project_root / ".env"
        if env_path.exists():
            self.log("Environment file already exists, skipping", "WARNING")
            return True

        env_content = """# Quantum Computing Environment Configuration

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=true

# Quantum Framework Settings
QISKIT_IBMQ_TOKEN=your_ibmq_token_here
GOOGLE_CLOUD_PROJECT=your_google_cloud_project
PENNYLANE_DEVICE=default.qubit

# Development Settings
PYTHONPATH=src
JUPYTER_PORT=8888

# Optional: Cloud Quantum Computing
# IBM_Q_HUB=your_hub
# IBM_Q_GROUP=your_group
# IBM_Q_PROJECT=your_project
"""

        try:
            with open(env_path, 'w') as f:
                f.write(env_content)

            self.log("Environment file created successfully", "SUCCESS")
            self.log(f"Edit {env_path} to configure your quantum computing accounts", "INFO")
            return True

        except Exception as e:
            self.log(f"Failed to create environment file: {e}", "ERROR")
            return False

    def print_setup_summary(self, validation_results: Dict[str, bool]):
        """Print comprehensive setup summary."""
        print("\n" + "="*60)
        print("QUANTUM COMPUTING ENVIRONMENT SETUP SUMMARY")
        print("="*60)

        print("📦 Installed Packages:")
        print(f"  ✅ Successfully installed: {len(self.installed_packages)}")
        if self.installed_packages:
            for package in self.installed_packages[:5]:  # Show first 5
                print(f"    - {package}")
            if len(self.installed_packages) > 5:
                print(f"    ... and {len(self.installed_packages) - 5} more")

        if self.failed_packages:
            print(f"  ❌ Failed to install: {len(self.failed_packages)}")
            for package in self.failed_packages:
                print(f"    - {package}")

        print("\n🔬 Framework Validation:")
        for framework, success in validation_results.items():
            status = "✅" if success else "❌"
            framework_name = framework.replace('_', ' ').title()
            print(f"  {status} {framework_name}")

        working_frameworks = sum(validation_results.values())
        total_frameworks = len(validation_results)

        print("
🚀 Getting Started:"        print("  1. Run: make dev")
        print("  2. Open: http://localhost:8888")
        print("  3. Try: notebooks/tutorials/quantum_basics.ipynb")

        print("
🧪 Testing:"        print("  Run: make test")
        print("  Run: make quantum-simulate")

        print("
📚 Resources:"        print("  - Qiskit: https://qiskit.org/documentation/")
        print("  - Cirq: https://quantumai.google/cirq")
        print("  - PennyLane: https://pennylane.ai/")

        if working_frameworks == total_frameworks:
            print(f"\n🎉 Setup completed successfully! All {working_frameworks} frameworks working.")
        elif working_frameworks > 0:
            print(f"\n⚠️  Setup completed with partial success. {working_frameworks}/{total_frameworks} frameworks working.")
        else:
            print("
❌ Setup failed. Check error messages above."            print("   Try installing packages manually or check Python version.")

        print("="*60)

    def run_complete_setup(self) -> bool:
        """Run the complete quantum environment setup."""
        print("🔬 QUANTUM COMPUTING ENVIRONMENT SETUP")
        print("=" * 50)

        success = True

        # Check Python version
        if not self.check_python_version():
            return False

        # Install quantum frameworks
        if not self.install_quantum_frameworks():
            self.log("Warning: No quantum frameworks installed", "WARNING")
            success = False

        # Install development tools
        if not self.install_development_tools():
            self.log("Warning: Some development tools failed to install", "WARNING")

        # Setup Jupyter kernel
        self.setup_jupyter_kernel()

        # Create environment file
        self.create_env_file()

        # Validate installation
        validation_results = self.validate_installation()

        # Print summary
        self.print_setup_summary(validation_results)

        # Check if basic functionality works
        basic_working = validation_results.get('project_imports', False)

        if basic_working:
            self.log("Quantum computing environment is ready! 🚀", "SUCCESS")
        else:
            self.log("Setup completed but some components may not work correctly", "WARNING")

        return basic_working


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Quantum Computing Environment Setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-validation", action="store_true", help="Skip final validation")
    parser.add_argument("--only-validate", action="store_true", help="Only run validation")

    args = parser.parse_args()

    setup = QuantumEnvironmentSetup(verbose=args.verbose)

    if args.only_validate:
        validation_results = setup.validate_installation()
        setup.print_setup_summary(validation_results)
        return 0 if all(validation_results.values()) else 1

    success = setup.run_complete_setup()

    if args.skip_validation:
        return 0

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
