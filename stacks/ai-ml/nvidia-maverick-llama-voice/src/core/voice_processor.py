"""
NVIDIA Maverick Voice Processor

This module provides comprehensive voice processing capabilities following NVIDIA's
Maverick architecture patterns for Llama 4 Voice model integration.
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import io

import numpy as np
import torch
import torchaudio
from transformers import pipeline
import soundfile as sf
import librosa
from scipy import signal
from scipy.io import wavfile

from .model_manager import ModelManager, ModelType, ModelConfig, create_speech_config, create_tts_config


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    PCM = "pcm"


class VoiceEmotion(Enum):
    """Supported voice emotions for synthesis."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CALM = "calm"
    EXCITED = "excited"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    format: AudioFormat = AudioFormat.WAV
    chunk_size: int = 1024
    buffer_size: int = 4096


@dataclass
class VoiceFeatures:
    """Extracted voice features for model input."""
    mfcc: np.ndarray
    pitch: np.ndarray
    energy: np.ndarray
    spectral_centroid: np.ndarray
    chroma: np.ndarray
    emotion_probabilities: Dict[str, float]
    embedding: np.ndarray


@dataclass
class VoiceProcessingResult:
    """Result of voice processing operations."""
    text: str = ""
    audio_data: Optional[np.ndarray] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    voice_features: Optional[VoiceFeatures] = None
    emotion: VoiceEmotion = VoiceEmotion.NEUTRAL
    language: str = "en-US"


class AudioStream:
    """Audio streaming interface for real-time processing."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.is_active = False
        self.stream_thread: Optional[threading.Thread] = None

    def start_stream(self):
        """Start audio streaming."""
        self.is_active = True
        self.stream_thread = threading.Thread(target=self._stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def stop_stream(self):
        """Stop audio streaming."""
        self.is_active = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)

    def _stream_worker(self):
        """Audio streaming worker thread."""
        # This would integrate with actual audio hardware
        # For demonstration, we'll simulate audio streaming
        try:
            while self.is_active:
                # Simulate audio chunk
                chunk = np.random.randn(self.config.chunk_size).astype(np.float32)
                self.audio_queue.put(chunk)
                time.sleep(self.config.chunk_size / self.config.sample_rate)
        except Exception as e:
            logging.error(f"Audio streaming error: {str(e)}")

    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get next audio chunk from stream."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def async_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """Async generator for audio streaming."""
        loop = asyncio.get_event_loop()

        while self.is_active:
            chunk = await loop.run_in_executor(None, self.get_audio_chunk, 0.1)
            if chunk is not None:
                yield chunk
            else:
                await asyncio.sleep(0.01)


class VoiceProcessor:
    """
    Main voice processing engine following NVIDIA Maverick patterns.

    This class provides comprehensive voice I/O capabilities including:
    - Real-time speech recognition
    - High-quality voice synthesis
    - Voice activity detection
    - Audio enhancement and noise reduction
    - Emotion recognition and adaptation
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.model_manager = ModelManager()
        self.audio_config = AudioConfig(
            sample_rate=self.config.get("sample_rate", 16000),
            channels=self.config.get("channels", 1)
        )

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self._speech_recognizer = None
        self._voice_synthesizer = None
        self._emotion_recognizer = None
        self._audio_enhancer = None

        # Audio processing state
        self.audio_stream: Optional[AudioStream] = None
        self.is_initialized = False

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load voice processing configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        # Default configuration
        return {
            "speech_recognition": {
                "model": "nvidia/parakeet-ctc-1.1b",
                "language": "en-US",
                "vad_threshold": 0.5
            },
            "voice_synthesis": {
                "model": "nvidia/tts_en_fastpitch",
                "voice": "ljspeech",
                "sample_rate": 22050
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024
            },
            "emotion_recognition": {
                "enabled": True,
                "model": "emotion_recognition"
            }
        }

    async def initialize(self):
        """Initialize voice processing components."""
        try:
            self.logger.info("Initializing NVIDIA Maverick Voice Processor")

            # Initialize speech recognition model
            speech_config = create_speech_config(
                "speech_recognition",
                self.config["speech_recognition"]["model"]
            )
            await self.model_manager.load_model("speech_recognition")

            # Initialize voice synthesis model
            tts_config = create_tts_config(
                "voice_synthesis",
                self.config["voice_synthesis"]["model"]
            )
            await self.model_manager.load_model("voice_synthesis")

            # Initialize audio stream
            self.audio_stream = AudioStream(self.audio_config)

            self.is_initialized = True
            self.logger.info("Voice processor initialization complete")

        except Exception as e:
            self.logger.error(f"Voice processor initialization failed: {str(e)}")
            raise

    async def speech_to_text(self, audio_data: Union[np.ndarray, str, Path],
                           language: Optional[str] = None) -> VoiceProcessingResult:
        """
        Convert speech audio to text.

        Args:
            audio_data: Audio data as numpy array, file path, or bytes
            language: Language code (optional)

        Returns:
            VoiceProcessingResult with transcribed text
        """
        start_time = time.time()

        try:
            # Load audio data
            audio_array = self._load_audio_data(audio_data)

            # Voice activity detection
            if not self._detect_voice_activity(audio_array):
                return VoiceProcessingResult(
                    text="",
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )

            # Audio enhancement
            enhanced_audio = await self._enhance_audio(audio_array)

            # Extract voice features
            voice_features = await self._extract_voice_features(enhanced_audio)

            # Speech recognition
            recognition_result = await self.model_manager.inference(
                "speech_recognition",
                {
                    "audio": enhanced_audio,
                    "sample_rate": self.audio_config.sample_rate,
                    "language": language or self.config["speech_recognition"]["language"]
                }
            )

            # Emotion recognition
            emotion = await self._recognize_emotion(voice_features)

            return VoiceProcessingResult(
                text=recognition_result.get("text", ""),
                confidence=recognition_result.get("confidence", 0.0),
                processing_time=time.time() - start_time,
                voice_features=voice_features,
                emotion=emotion,
                language=language or self.config["speech_recognition"]["language"]
            )

        except Exception as e:
            self.logger.error(f"Speech-to-text failed: {str(e)}")
            return VoiceProcessingResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time
            )

    async def text_to_speech(self, text: str, voice: Optional[str] = None,
                           emotion: Optional[VoiceEmotion] = None) -> VoiceProcessingResult:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize
            voice: Voice to use (optional)
            emotion: Emotional tone (optional)

        Returns:
            VoiceProcessingResult with synthesized audio
        """
        start_time = time.time()

        try:
            # Prepare synthesis parameters
            voice_name = voice or self.config["voice_synthesis"]["voice"]
            target_emotion = emotion or VoiceEmotion.NEUTRAL

            # Adjust parameters based on emotion
            emotion_params = self._get_emotion_parameters(target_emotion)

            # Voice synthesis
            synthesis_result = await self.model_manager.inference(
                "voice_synthesis",
                {
                    "text": text,
                    "voice": voice_name,
                    "emotion": target_emotion.value,
                    "sample_rate": self.config["voice_synthesis"]["sample_rate"],
                    **emotion_params
                }
            )

            audio_data = synthesis_result.get("audio")
            if audio_data is not None:
                # Audio post-processing
                processed_audio = await self._postprocess_audio(audio_data)

                return VoiceProcessingResult(
                    text=text,
                    audio_data=processed_audio,
                    confidence=synthesis_result.get("confidence", 1.0),
                    processing_time=time.time() - start_time,
                    emotion=target_emotion
                )
            else:
                raise ValueError("Voice synthesis failed to generate audio")

        except Exception as e:
            self.logger.error(f"Text-to-speech failed: {str(e)}")
            return VoiceProcessingResult(
                text=text,
                confidence=0.0,
                processing_time=time.time() - start_time
            )

    async def stream_speech_to_text(self, audio_chunk: np.ndarray) -> str:
        """
        Real-time speech-to-text for streaming audio.

        Args:
            audio_chunk: Audio chunk as numpy array

        Returns:
            Transcribed text chunk
        """
        try:
            # Process audio chunk
            enhanced_chunk = await self._enhance_audio_chunk(audio_chunk)

            # Streaming recognition
            result = await self.model_manager.inference(
                "speech_recognition",
                {
                    "audio_chunk": enhanced_chunk,
                    "streaming": True,
                    "sample_rate": self.audio_config.sample_rate
                }
            )

            return result.get("text", "")

        except Exception as e:
            self.logger.error(f"Streaming speech recognition failed: {str(e)}")
            return ""

    async def stream_text_to_speech(self, text_chunk: str,
                                  emotion: VoiceEmotion = VoiceEmotion.NEUTRAL) -> np.ndarray:
        """
        Real-time text-to-speech for streaming text.

        Args:
            text_chunk: Text chunk to synthesize
            emotion: Emotional tone

        Returns:
            Synthesized audio chunk
        """
        try:
            # Streaming synthesis
            result = await self.model_manager.inference(
                "voice_synthesis",
                {
                    "text_chunk": text_chunk,
                    "streaming": True,
                    "emotion": emotion.value,
                    "sample_rate": self.audio_config.sample_rate
                }
            )

            return result.get("audio_chunk", np.array([]))

        except Exception as e:
            self.logger.error(f"Streaming voice synthesis failed: {str(e)}")
            return np.array([])

    def listen(self) -> AudioStream:
        """Start audio listening stream."""
        if not self.audio_stream:
            self.audio_stream = AudioStream(self.audio_config)

        if not self.audio_stream.is_active:
            self.audio_stream.start_stream()

        return self.audio_stream

    def stop_listening(self):
        """Stop audio listening stream."""
        if self.audio_stream and self.audio_stream.is_active:
            self.audio_stream.stop_stream()

    async def play_audio(self, audio_data: np.ndarray, sample_rate: Optional[int] = None):
        """Play audio data."""
        try:
            sr = sample_rate or self.config["voice_synthesis"]["sample_rate"]

            # This would integrate with audio playback hardware
            # For demonstration, we'll just log the action
            self.logger.info(".1f"
        except Exception as e:
            self.logger.error(f"Audio playback failed: {str(e)}")

    async def save_audio(self, audio_data: np.ndarray, filename: str,
                        sample_rate: Optional[int] = None):
        """Save audio data to file."""
        try:
            sr = sample_rate or self.config["voice_synthesis"]["sample_rate"]
            filepath = Path(filename)

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save audio file
            sf.write(str(filepath), audio_data, sr)
            self.logger.info(f"Audio saved to: {filepath}")

        except Exception as e:
            self.logger.error(f"Audio save failed: {str(e)}")

    def _load_audio_data(self, audio_input: Union[np.ndarray, str, Path, bytes]) -> np.ndarray:
        """Load audio data from various input formats."""
        if isinstance(audio_input, np.ndarray):
            return audio_input
        elif isinstance(audio_input, (str, Path)):
            # Load from file
            audio, _ = librosa.load(str(audio_input), sr=self.audio_config.sample_rate)
            return audio
        elif isinstance(audio_input, bytes):
            # Load from bytes
            audio, _ = sf.read(io.BytesIO(audio_input))
            return audio
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Detect voice activity in audio data."""
        # Simple energy-based VAD
        energy = np.mean(audio_data ** 2)
        threshold = self.config["speech_recognition"]["vad_threshold"]

        return energy > threshold

    async def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance audio quality with noise reduction."""
        # Apply noise reduction if configured
        if self.config.get("noise_reduction", True):
            # Simple noise gate
            noise_threshold = np.percentile(np.abs(audio_data), 10)
            enhanced = np.where(np.abs(audio_data) > noise_threshold, audio_data, 0)
            return enhanced

        return audio_data

    async def _enhance_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Enhance a chunk of audio data."""
        # Simplified enhancement for streaming
        return audio_chunk * 1.2  # Simple amplification

    async def _extract_voice_features(self, audio_data: np.ndarray) -> VoiceFeatures:
        """Extract comprehensive voice features."""
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.audio_config.sample_rate, n_mfcc=13)

        # Pitch estimation
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.audio_config.sample_rate)
        pitch = pitches[magnitudes > np.median(magnitudes)]

        # Energy
        energy = librosa.feature.rms(y=audio_data)

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.audio_config.sample_rate)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.audio_config.sample_rate)

        # Emotion probabilities (simplified)
        emotion_probs = {
            "neutral": 0.4,
            "happy": 0.3,
            "sad": 0.2,
            "angry": 0.1
        }

        # Voice embedding (simplified)
        embedding = np.random.randn(768).astype(np.float32)  # Mock embedding

        return VoiceFeatures(
            mfcc=mfcc,
            pitch=pitch if len(pitch) > 0 else np.array([]),
            energy=energy,
            spectral_centroid=spectral_centroid,
            chroma=chroma,
            emotion_probabilities=emotion_probs,
            embedding=embedding
        )

    async def _recognize_emotion(self, voice_features: VoiceFeatures) -> VoiceEmotion:
        """Recognize emotion from voice features."""
        if not self.config["emotion_recognition"]["enabled"]:
            return VoiceEmotion.NEUTRAL

        # Simple emotion recognition based on features
        emotion_scores = voice_features.emotion_probabilities
        best_emotion = max(emotion_scores, key=emotion_scores.get)

        try:
            return VoiceEmotion(best_emotion)
        except ValueError:
            return VoiceEmotion.NEUTRAL

    def _get_emotion_parameters(self, emotion: VoiceEmotion) -> Dict[str, Any]:
        """Get synthesis parameters for emotion."""
        emotion_params = {
            VoiceEmotion.NEUTRAL: {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
            VoiceEmotion.HAPPY: {"speed": 1.1, "pitch": 1.2, "energy": 1.3},
            VoiceEmotion.SAD: {"speed": 0.9, "pitch": 0.8, "energy": 0.7},
            VoiceEmotion.ANGRY: {"speed": 1.2, "pitch": 1.3, "energy": 1.4},
            VoiceEmotion.SURPRISED: {"speed": 1.3, "pitch": 1.4, "energy": 1.5},
            VoiceEmotion.CALM: {"speed": 0.8, "pitch": 0.9, "energy": 0.6},
            VoiceEmotion.EXCITED: {"speed": 1.4, "pitch": 1.3, "energy": 1.6}
        }

        return emotion_params.get(emotion, emotion_params[VoiceEmotion.NEUTRAL])

    async def _postprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Post-process synthesized audio."""
        # Apply audio normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.8  # Normalize to 80%

        return audio_data

    async def get_voice_metrics(self) -> Dict[str, Any]:
        """Get voice processing performance metrics."""
        model_metrics = {}
        for model_name in ["speech_recognition", "voice_synthesis"]:
            metrics = self.model_manager.get_model_metrics(model_name)
            if metrics:
                model_metrics[model_name] = {
                    "inference_time": metrics.inference_time,
                    "latency_p95": metrics.latency_p95,
                    "throughput": metrics.throughput
                }

        return {
            "models": model_metrics,
            "audio_config": {
                "sample_rate": self.audio_config.sample_rate,
                "channels": self.audio_config.channels,
                "chunk_size": self.audio_config.chunk_size
            },
            "capabilities": {
                "speech_recognition": True,
                "voice_synthesis": True,
                "emotion_recognition": self.config["emotion_recognition"]["enabled"],
                "streaming": True,
                "noise_reduction": True
            }
        }


# Context manager support
class voice_processor:
    """Context manager for voice processing operations."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.processor: Optional[VoiceProcessor] = None

    async def __aenter__(self):
        self.processor = VoiceProcessor(self.config_path)
        await self.processor.initialize()
        return self.processor

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.processor:
            self.processor.stop_listening()


# Convenience functions
async def speech_to_text_file(audio_file: str, config_path: Optional[str] = None) -> str:
    """Convert audio file to text."""
    async with voice_processor(config_path) as vp:
        result = await vp.speech_to_text(audio_file)
        return result.text


async def text_to_speech_file(text: str, output_file: str,
                            config_path: Optional[str] = None,
                            voice: Optional[str] = None):
    """Convert text to speech and save to file."""
    async with voice_processor(config_path) as vp:
        result = await vp.text_to_speech(text, voice)
        if result.audio_data is not None:
            await vp.save_audio(result.audio_data, output_file)


def create_voice_config(sample_rate: int = 16000,
                       language: str = "en-US",
                       voice: str = "ljspeech") -> Dict[str, Any]:
    """Create a voice processing configuration."""
    return {
        "speech_recognition": {
            "model": "nvidia/parakeet-ctc-1.1b",
            "language": language,
            "vad_threshold": 0.5
        },
        "voice_synthesis": {
            "model": "nvidia/tts_en_fastpitch",
            "voice": voice,
            "sample_rate": 22050
        },
        "audio": {
            "sample_rate": sample_rate,
            "channels": 1,
            "chunk_size": 1024
        },
        "emotion_recognition": {
            "enabled": True,
            "model": "emotion_recognition"
        },
        "noise_reduction": True,
        "echo_cancellation": True
    }
