"""
Setup script for NVIDIA Maverick Llama Voice Model.

This setup script configures the nvidia-maverick-llama-voice package for installation
and distribution of the voice-enabled AI model.
"""

from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="nvidia-maverick-llama-voice",
    version="1.0.0",
    author="Template Heaven Team",
    author_email="team@templateheaven.org",
    description="NVIDIA Maverick Llama 4 Voice Model - Voice-enabled AI following NVIDIA architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/template-heaven/nvidia-maverick-llama-voice",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache-2.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Communications",
    ],
    keywords=[
        "ai", "voice", "speech", "recognition", "synthesis", "nvidia",
        "maverick", "llama", "voice-model", "conversational-ai",
        "real-time", "streaming", "emotion-recognition"
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "transformers>=4.21.0",
        "datasets>=2.4.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "asyncio-mqtt>=0.11.0",
        "websockets>=10.0",
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
    ],
    extras_require={
        "nvidia": [
            "nvidia-pyindex",
            "nvidia-tensorrt>=8.4.0",
            "nvidia-tritonserver>=2.24.0",
            "nvidia-riva>=2.8.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
            "jax>=0.3.0",
            "jaxlib>=0.3.0",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "web": [
            "flask>=2.1.0",
            "flask-socketio>=5.1.0",
            "python-socketio>=5.5.0",
        ],
        "all": [
            "nvidia-maverick-llama-voice[nvidia,gpu,dev,web]",
        ],
    },
    entry_points={
        "console_scripts": [
            "maverick-voice=nvidia_maverick_llama_voice.cli:main",
            "maverick-chat=nvidia_maverick_llama_voice.scripts.voice_chat:main",
        ],
    },
    package_data={
        "nvidia_maverick_llama_voice": [
            "config/*.yaml",
            "config/*.yml",
            "web/static/*",
            "web/templates/*",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    scripts=[
        "run_voice.py",
        "scripts/voice_chat.py",
    ],
    project_urls={
        "Homepage": "https://github.com/template-heaven/nvidia-maverick-llama-voice",
        "Documentation": "https://nvidia-maverick-llama-voice.readthedocs.io/",
        "Repository": "https://github.com/template-heaven/nvidia-maverick-llama-voice",
        "Issues": "https://github.com/template-heaven/nvidia-maverick-llama-voice/issues",
        "NVIDIA": "https://www.nvidia.com/en-us/ai/",
    },
)
