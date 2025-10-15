"""
Setup script for Quantum Computing Starter Template.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quantum-computing-starter",
    version="1.0.0",
    description="Quantum computing development environment with Qiskit, Cirq, and PennyLane",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Template Heaven Team",
    author_email="team@templateheaven.dev",
    url="https://github.com/template-heaven/templateheaven",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
        ],
        "cloud": [
            "qiskit-ibmq-provider>=0.20.0",
            "cirq-google>=1.0.0",
            # "amazon-braket-sdk>=1.0.0",  # Uncomment when available
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="quantum computing qiskit cirq pennylane machine learning",
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "quantum-starter=quantum_computing_starter.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/template-heaven/templateheaven/issues",
        "Source": "https://github.com/template-heaven/templateheaven",
        "Documentation": "https://templateheaven.dev/docs/quantum-computing",
    },
)
