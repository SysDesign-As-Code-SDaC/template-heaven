#!/usr/bin/env python3
"""
NVIDIA Maverick Llama 4 Voice Model - Main Entry Point

This is the primary entry point for running the NVIDIA Maverick Llama 4 Voice model.
Provides a unified interface for voice-enabled AI interactions following NVIDIA's architecture.

Usage:
    python run_voice.py chat              # Start voice chat
    python run_voice.py text              # Start text chat
    python run_voice.py benchmark         # Run performance benchmarks
    python run_voice.py setup             # Setup and configure
    python run_voice.py --help            # Show help
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.voice_processor import VoiceProcessor, create_voice_config
from core.session_manager import SessionManager
from core.inference_engine import create_voice_inference_engine
import scripts.voice_chat as voice_chat


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nvidia_maverick.log')
        ]
    )


async def run_voice_chat(args):
    """Run voice chat interface."""
    print("🎤 Starting NVIDIA Maverick Voice Chat...")
    print("=" * 50)

    chat = voice_chat.VoiceChat(
        session_id=getattr(args, 'session', None),
        language=getattr(args, 'language', 'en-US'),
        personality=getattr(args, 'personality', 'friendly')
    )

    try:
        await chat.initialize()
        await chat.start_conversation()
    except Exception as e:
        print(f"❌ Voice chat failed: {str(e)}")
        return 1

    return 0


async def run_text_chat(args):
    """Run text chat interface."""
    print("💬 Starting NVIDIA Maverick Text Chat...")
    print("=" * 50)

    chat = voice_chat.TextChat(
        session_id=getattr(args, 'session', None),
        language=getattr(args, 'language', 'en-US'),
        personality=getattr(args, 'personality', 'friendly')
    )

    try:
        await chat.initialize()
        await chat.start_conversation()
    except Exception as e:
        print(f"❌ Text chat failed: {str(e)}")
        return 1

    return 0


async def run_benchmark(args):
    """Run performance benchmarks."""
    print("📊 Running NVIDIA Maverick Benchmarks...")
    print("=" * 50)

    try:
        # Import benchmark script
        import scripts.benchmark_voice as benchmark
        await benchmark.run_benchmarks(args)
    except ImportError:
        print("❌ Benchmark script not found")
        return 1
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        return 1

    return 0


async def run_setup(args):
    """Run setup and configuration."""
    print("🔧 Setting up NVIDIA Maverick Llama Voice...")
    print("=" * 50)

    try:
        # Check system requirements
        print("📋 Checking system requirements...")

        # Check Python version
        if sys.version_info < (3, 9):
            print("❌ Python 3.9+ required")
            return 1
        print("✅ Python version: OK")

        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"✅ NVIDIA GPU detected: {gpu_count} device(s)")
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    print(f"  - GPU {i}: {name}")
            else:
                print("⚠️  No NVIDIA GPU detected (CPU mode will be used)")
        except ImportError:
            print("⚠️  PyTorch not installed")

        # Create configuration
        print("\n⚙️  Creating default configuration...")
        config = create_voice_config()
        config_path = Path("config/voice_config.yaml")

        import yaml
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✅ Configuration saved to: {config_path}")

        # Initialize components
        print("\n🚀 Initializing components...")

        # Test voice processor initialization
        voice_config = create_voice_config()
        processor = VoiceProcessor()
        await processor.initialize()
        print("✅ Voice processor: OK")

        # Test session manager
        session_mgr = SessionManager()
        session_id = await session_mgr.create_session()
        print(f"✅ Session manager: OK (session: {session_id})")

        # Test inference engine
        engine = await create_voice_inference_engine()
        await engine.initialize()
        print("✅ Inference engine: OK")

        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Run 'python run_voice.py chat' for voice chat")
        print("2. Run 'python run_voice.py text' for text chat")
        print("3. Run 'python run_voice.py benchmark' for performance tests")

    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        return 1

    return 0


async def run_demo(args):
    """Run a quick demo."""
    print("🎭 NVIDIA Maverick Demo")
    print("=" * 30)

    try:
        # Quick text interaction
        session_mgr = SessionManager()
        session_id = await session_mgr.create_session(personality="friendly")

        print("🤖 AI: Hello! I'm your NVIDIA Maverick assistant. How can I help you today?")
        print()

        # Simple interaction loop
        for i in range(3):
            user_input = input(f"👤 You ({i+1}/3): ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                break

            response = await session_mgr.process_message(session_id, user_input)
            print(f"🤖 AI: {response['response']}")
            print()

        print("👋 Demo complete! Run 'python run_voice.py chat' for full voice interaction.")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return 1

    return 0


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="NVIDIA Maverick Llama 4 Voice Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NVIDIA Maverick Llama 4 Voice Model - Complete Voice-Enabled AI Solution

COMMANDS:
  chat        Start interactive voice chat
  text        Start interactive text chat
  benchmark   Run performance benchmarks
  setup       Setup and configure the system
  demo        Run a quick demo

EXAMPLES:
  python run_voice.py chat                    # Voice chat
  python run_voice.py text                    # Text chat
  python run_voice.py benchmark               # Performance tests
  python run_voice.py setup                   # Initial setup
  python run_voice.py demo                    # Quick demo

ADVANCED USAGE:
  python run_voice.py chat --language es-ES   # Spanish voice chat
  python run_voice.py text --personality technical  # Technical AI personality
  python run_voice.py benchmark --extended    # Extended benchmarks

For more information, visit: https://nvidia.com/maverick
        """
    )

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start voice chat')
    chat_parser.add_argument('--session', type=str,
                            help='Resume existing session ID')
    chat_parser.add_argument('--language', type=str, default='en-US',
                            help='Language code (default: en-US)')
    chat_parser.add_argument('--personality', type=str, default='friendly',
                            choices=['neutral', 'friendly', 'professional', 'creative', 'technical'],
                            help='AI personality (default: friendly)')

    # Text command
    text_parser = subparsers.add_parser('text', help='Start text chat')
    text_parser.add_argument('--session', type=str,
                            help='Resume existing session ID')
    text_parser.add_argument('--language', type=str, default='en-US',
                            help='Language code (default: en-US)')
    text_parser.add_argument('--personality', type=str, default='friendly',
                            choices=['neutral', 'friendly', 'professional', 'creative', 'technical'],
                            help='AI personality (default: friendly)')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--extended', action='store_true',
                                 help='Run extended benchmark suite')
    benchmark_parser.add_argument('--output', type=str, default='benchmark_results',
                                 help='Output directory for results')

    # Setup command
    subparsers.add_parser('setup', help='Setup and configure')

    # Demo command
    subparsers.add_parser('demo', help='Run quick demo')

    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.debug:
        setup_logging(logging.DEBUG)
    elif args.verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    try:
        if args.command == 'chat':
            return await run_voice_chat(args)
        elif args.command == 'text':
            return await run_text_chat(args)
        elif args.command == 'benchmark':
            return await run_benchmark(args)
        elif args.command == 'setup':
            return await run_setup(args)
        elif args.command == 'demo':
            return await run_demo(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
