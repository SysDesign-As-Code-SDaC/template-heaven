#!/usr/bin/env python3
"""
Setup script for Template Heaven package.

This file provides backward compatibility for older pip versions
that don't support pyproject.toml. The main configuration is in pyproject.toml.
"""

from setuptools import setup

# Read version from __version__.py
import os
import sys

# Add the package directory to the path to import __version__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'templateheaven'))

try:
    from __version__ import __version__
except ImportError:
    __version__ = "0.1.0"

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Interactive template management for Template Heaven"

setup(
    name="templateheaven",
    version=__version__,
    description="Interactive template management for Template Heaven",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Template Heaven Team",
    author_email="team@templateheaven.dev",
    url="https://github.com/template-heaven/templateheaven",
    project_urls={
        "Homepage": "https://github.com/template-heaven/templateheaven",
        "Documentation": "https://templateheaven.dev/docs",
        "Repository": "https://github.com/template-heaven/templateheaven.git",
        "Issues": "https://github.com/template-heaven/templateheaven/issues",
        "Changelog": "https://github.com/template-heaven/templateheaven/blob/main/CHANGELOG.md",
    },
    packages=["templateheaven"],
    package_data={
        "templateheaven": ["data/*.yaml", "data/templates/*"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0",
        "rich>=13.0",
        "questionary>=2.0",
        "pyyaml>=6.0",
        "jinja2>=3.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.10",
            "black>=23.0",
            "mypy>=1.0",
            "flake8>=6.0",
            "isort>=5.12",
        ],
        "ui": [
            "streamlit>=1.28",
        ],
        "github": [
            "gitpython>=3.1",
            "requests>=2.28",
            "aiohttp>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "templateheaven=templateheaven.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Utilities",
    ],
    keywords="templates boilerplate project-starter cli development",
    zip_safe=False,
)
