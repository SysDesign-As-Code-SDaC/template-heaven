"""
Setup script for AI Benchmarks Suite.

This setup script configures the ai-benchmarks package for installation
and distribution.
"""

from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-benchmarks-suite",
    version="1.0.0",
    author="Template Heaven Team",
    author_email="team@templateheaven.org",
    description="Comprehensive AI Benchmarks Suite for evaluating AI systems across all intelligence paradigms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/template-heaven/ai-benchmarks-suite",
    packages=find_packages(where="ai_benchmarks"),
    package_dir={"": "ai_benchmarks"},
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Benchmark",
    ],
    keywords=[
        "ai", "benchmarking", "machine-learning", "artificial-intelligence",
        "neuromorphic", "quantum-computing", "agi", "asi", "evaluation",
        "performance-testing", "ai-safety", "interpretability"
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "neuromorphic": [
            "norse>=0.0.7",
            "snnTorch>=0.4.0",
        ],
        "quantum": [
            "qiskit>=0.30.0",
            "pennylane>=0.15.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
            "jax>=0.2.0",
            "jaxlib>=0.1.0",
        ],
        "multimodal": [
            "Pillow>=8.0.0",
            "opencv-python>=4.5.0",
            "librosa>=0.8.0",
            "nltk>=3.6.0",
            "spacy>=3.2.0",
        ],
        "tracking": [
            "mlflow>=1.20.0",
            "wandb>=0.12.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
        "dev": [
            "black>=21.0.0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
            "pytest>=6.2.0",
            "pytest-asyncio>=0.15.0",
            "pytest-cov>=2.12.0",
        ],
        "all": [
            "ai-benchmarks-suite[neuromorphic,quantum,gpu,multimodal,tracking,docs,dev]",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-benchmarks=ai_benchmarks.cli:main",
        ],
    },
    package_data={
        "ai_benchmarks": [
            "config/*.yaml",
            "config/*.yml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Homepage": "https://github.com/template-heaven/ai-benchmarks-suite",
        "Documentation": "https://ai-benchmarks-suite.readthedocs.io/",
        "Repository": "https://github.com/template-heaven/ai-benchmarks-suite",
        "Issues": "https://github.com/template-heaven/ai-benchmarks-suite/issues",
        "Changelog": "https://github.com/template-heaven/ai-benchmarks-suite/blob/main/CHANGELOG.md",
    },
)
