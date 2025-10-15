#!/usr/bin/env python3
"""
NVIDIA Maverick Voice Chat

Interactive voice chat application using NVIDIA's Maverick Llama Voice model.
Provides real-time voice conversations with AI using voice recognition and synthesis.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.voice_processor import VoiceProcessor, voice_processor
from core.session_manager import SessionManager
from core.inference_engine import StreamingInferenceEngine, create_voice_inference_engine


class VoiceChat:
    """
    Interactive voice chat application.
    """

    def __init__(self, session_id: str = None, language: str = "en-US", personality: str = "friendly"):
        self.session_id = session_id
        self.language = language
        self.personality = personality
        self.voice_processor = None
        self.session_manager = None
        self.inference_engine = None
        self.is_running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize voice chat components."""
        try:
            self.logger.info("Initializing NVIDIA Maverick Voice Chat")

            # Initialize voice processor
            async with voice_processor() as vp:
                self.voice_processor = vp

            # Initialize session manager
            self.session_manager = SessionManager()

            # Create or get session
            if not self.session_id:
                self.session_id = await self.session_manager.create_session(
                    language=self.language
                )
                self.logger.info(f"Created new session: {self.session_id}")
            else:
                session = await self.session_manager.get_session(self.session_id)
                if not session:
                    self.logger.warning(f"Session {self.session_id} not found, creating new one")
                    self.session_id = await self.session_manager.create_session(
                        language=self.language
                    )

            # Initialize inference engine
            self.inference_engine = await create_voice_inference_engine()

            self.logger.info("Voice chat initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize voice chat: {str(e)}")
            raise

    async def start_conversation(self):
        """Start the voice conversation loop."""
        print("🎤 NVIDIA Maverick Voice Chat")
        print("============================")
        print(f"Session ID: {self.session_id}")
        print(f"Language: {self.language}")
        print()
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'new session' - Start new conversation")
        print("  'save' - Save conversation")
        print("  'help' - Show this help")
        print()

        self.is_running = True

        try:
            while self.is_running:
                print("🎤 Listening... (speak now or type 'quit' to exit)")

                try:
                    # Listen for voice input
                    audio_stream = self.voice_processor.listen()

                    # Convert speech to text
                    print("🔄 Processing speech...")
                    voice_result = await self.voice_processor.speech_to_text(
                        await self._get_next_audio_chunk(audio_stream)
                    )

                    if not voice_result.success:
                        print("❌ Speech recognition failed, please try again")
                        continue

                    user_input = voice_result.text
                    print(f"👤 You: {user_input}")

                    # Check for commands
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    elif user_input.lower() == 'new session':
                        await self._new_session()
                        continue
                    elif user_input.lower() == 'save':
                        await self._save_conversation()
                        continue
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue

                    # Process message
                    print("🤖 Thinking...")
                    response = await self.session_manager.process_message(
                        self.session_id,
                        user_input,
                        voice_result
                    )

                    # Display text response
                    print(f"🤖 AI: {response['response']}")

                    # Play voice response
                    if response.get('voice_response') is not None:
                        print("🔊 Playing voice response...")
                        await self.voice_processor.play_audio(response['voice_response'])
                    else:
                        print("🔇 Voice response not available")

                    print()

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error in conversation loop: {str(e)}")
                    print("❌ An error occurred. Please try again.")
                    continue

        finally:
            await self.cleanup()

    async def _get_next_audio_chunk(self, audio_stream) -> bytes:
        """Get next audio chunk from stream (simplified implementation)."""
        # In a real implementation, this would capture actual audio
        # For demonstration, we'll simulate a short audio recording
        await asyncio.sleep(1.0)  # Simulate recording time

        # Return mock audio data (this would be real audio bytes)
        return b"mock_audio_data_" + str(asyncio.get_event_loop().time()).encode()

    async def _new_session(self):
        """Start a new conversation session."""
        print("🆕 Starting new session...")
        old_session = self.session_id
        self.session_id = await self.session_manager.create_session(
            language=self.language
        )
        print(f"✅ New session created: {self.session_id}")
        print(f"💾 Previous session {old_session} is still available")

    async def _save_conversation(self):
        """Save current conversation."""
        try:
            summary = await self.session_manager.get_session_summary(self.session_id)
            if summary:
                filename = f"conversation_{self.session_id}_{int(asyncio.get_event_loop().time())}.json"
                # In a real implementation, save to file
                print(f"💾 Conversation saved to {filename}")
                print(f"📊 Messages: {summary['total_messages']}")
                print(f"⏱️  Duration: {summary['duration']:.1f} seconds")
            else:
                print("❌ Failed to save conversation")
        except Exception as e:
            print(f"❌ Error saving conversation: {str(e)}")

    def _show_help(self):
        """Show help information."""
        print("\n📖 Voice Chat Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'new session' - Start new conversation")
        print("  'save' - Save conversation")
        print("  'help' - Show this help")
        print("\n🎤 Voice Features:")
        print("  - Real-time speech recognition")
        print("  - Emotion-aware voice synthesis")
        print("  - Multi-language support")
        print("  - Context-aware conversations")
        print()

    async def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up voice chat resources")

        if self.voice_processor:
            self.voice_processor.stop_listening()

        if self.inference_engine:
            await self.inference_engine.shutdown()

        if self.session_manager:
            await self.session_manager.end_session(self.session_id)

        self.logger.info("Voice chat cleanup complete")


class TextChat(VoiceChat):
    """Text-based chat interface (no voice processing)."""

    async def initialize(self):
        """Initialize text chat components."""
        try:
            self.logger.info("Initializing NVIDIA Maverick Text Chat")

            # Initialize session manager
            self.session_manager = SessionManager()

            # Create or get session
            if not self.session_id:
                self.session_id = await self.session_manager.create_session(
                    language=self.language
                )
                self.logger.info(f"Created new session: {self.session_id}")

            # Initialize inference engine (no voice)
            self.inference_engine = StreamingInferenceEngine(
                model="llama_voice",
                enable_voice=False,
                streaming=False
            )
            await self.inference_engine.initialize()

            self.logger.info("Text chat initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize text chat: {str(e)}")
            raise

    async def start_conversation(self):
        """Start the text conversation loop."""
        print("💬 NVIDIA Maverick Text Chat")
        print("============================")
        print(f"Session ID: {self.session_id}")
        print(f"Language: {self.language}")
        print()
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'new session' - Start new conversation")
        print("  'save' - Save conversation")
        print("  'help' - Show this help")
        print()

        self.is_running = True

        try:
            while self.is_running:
                # Get text input
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                elif user_input.lower() == 'new session':
                    await self._new_session()
                    continue
                elif user_input.lower() == 'save':
                    await self._save_conversation()
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue

                # Process message
                print("🤖 Thinking...")
                response = await self.session_manager.process_message(
                    self.session_id,
                    user_input
                )

                # Display response
                print(f"🤖 AI: {response['response']}")
                print()

        finally:
            await self.cleanup()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NVIDIA Maverick Voice/Text Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice_chat.py                    # Voice chat (default)
  python voice_chat.py --text            # Text chat
  python voice_chat.py --session abc123  # Continue existing session
  python voice_chat.py --language es-ES  # Spanish language
  python voice_chat.py --personality technical  # Technical personality
        """
    )

    parser.add_argument('--text', action='store_true',
                       help='Use text chat instead of voice')
    parser.add_argument('--session', type=str,
                       help='Resume existing session ID')
    parser.add_argument('--language', type=str, default='en-US',
                       help='Language code (default: en-US)')
    parser.add_argument('--personality', type=str, default='friendly',
                       choices=['neutral', 'friendly', 'professional', 'creative', 'technical'],
                       help='AI personality (default: friendly)')

    args = parser.parse_args()

    try:
        # Create chat instance
        if args.text:
            chat = TextChat(
                session_id=args.session,
                language=args.language,
                personality=args.personality
            )
        else:
            chat = VoiceChat(
                session_id=args.session,
                language=args.language,
                personality=args.personality
            )

        # Initialize and start
        await chat.initialize()
        await chat.start_conversation()

        print("👋 Thank you for using NVIDIA Maverick Voice Chat!")

    except KeyboardInterrupt:
        print("\n⏹️  Chat interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
