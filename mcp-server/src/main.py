#!/usr/bin/env python3
"""
Template Heaven MCP Server Entry Point

Main entry point for starting the Template Heaven MCP server.
This server provides unified access to all Template Heaven services
through the Model Context Protocol (MCP).
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config
from core.mcp_server import TemplateHeavenMCPServer


def setup_logging(config):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if config.log_format == 'console'
        else '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )


async def main():
    """Main server startup function."""
    # Get configuration
    config = get_config()

    # Setup logging
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting Template Heaven MCP Server")
    logger.info(f"Server version: 1.0.0")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Debug mode: {config.debug}")

    # Create and start server
    server = TemplateHeavenMCPServer(config)

    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(server.graceful_shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
