"""
Main entry point for AI Benchmarks Suite.

This module provides the primary command-line interface for running comprehensive
AI benchmarks across all intelligence paradigms.

Usage:
    python -m ai_benchmarks run --suite comprehensive
    python -m ai_benchmarks list
    python -m ai_benchmarks report --input results/ --output reports/
"""

import sys
import asyncio
from .cli import main

if __name__ == "__main__":
    # Run the CLI asynchronously
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
