"""
NVIDIA Maverick Model Manager

This module provides model-agnostic management following NVIDIA's Maverick architecture.
Supports dynamic model loading, optimization, and lifecycle management for voice-enabled LLMs.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import time
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype, np_to_triton_dtype


class ModelBackend(Enum):
    """Supported model backends following NVIDIA patterns."""
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    TRITON = "triton"
    ONNX = "onnx"


class ModelType(Enum):
    """Supported model types for voice processing."""
    LLM = "llm"  # Large Language Model
    SPEECH_RECOGNITION = "speech_recognition"
    VOICE_SYNTHESIS = "voice_synthesis"
    VOICE_CLONING = "voice_cloning"
    EMOTION_RECOGNITION = "emotion_recognition"


@dataclass
class ModelConfig:
    """Model configuration following NVIDIA conventions."""
    name: str
    type: ModelType
    backend: ModelBackend
    model_path: Union[str, Path]
    device: str = "auto"
    precision: str = "fp16"
    max_batch_size: int = 1
    max_sequence_length: int = 2048
    voice_embedding_dim: int = 768
    enable_streaming: bool = True
    quantization: str = "none"  # none, int8, int4
    triton_url: Optional[str] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Performance metrics for model monitoring."""
    inference_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    gpu_utilization: float = 0.0
    error_rate: float = 0.0


class NVIDIADeviceManager:
    """NVIDIA GPU device management following Maverick patterns."""

    def __init__(self):
        self.devices = {}
        self._initialize_devices()

    def _initialize_devices(self):
        """Initialize available NVIDIA devices."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

                self.devices[f"cuda:{i}"] = {
                    "name": device_name,
                    "memory_gb": device_memory,
                    "compute_capability": torch.cuda.get_device_capability(i),
                    "available": True
                }
        else:
            self.devices["cpu"] = {
                "name": "CPU",
                "memory_gb": 0,
                "compute_capability": None,
                "available": True
            }

    def get_optimal_device(self, model_config: ModelConfig) -> str:
        """Get optimal device for model based on requirements."""
        if model_config.device != "auto":
            return model_config.device

        # For voice-enabled models, prefer A100/H100 if available
        preferred_gpus = ["A100", "H100", "A6000", "RTX 3090", "RTX 4090"]

        for device_id, device_info in self.devices.items():
            if device_id.startswith("cuda") and device_info["available"]:
                device_name = device_info["name"]
                if any(gpu in device_name for gpu in preferred_gpus):
                    return device_id

        # Fallback to first available GPU or CPU
        for device_id, device_info in self.devices.items():
            if device_info["available"]:
                return device_id

        return "cpu"

    def get_device_memory(self, device: str) -> float:
        """Get available memory for device in GB."""
        if device in self.devices:
            return self.devices[device]["memory_gb"]
        return 0.0


class BaseModelWrapper:
    """Base class for model wrappers following NVIDIA patterns."""

    def __init__(self, config: ModelConfig, device_manager: NVIDIADeviceManager):
        self.config = config
        self.device_manager = device_manager
        self.device = self.device_manager.get_optimal_device(config)
        self.model = None
        self.metrics = ModelMetrics()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance monitoring
        self._inference_times = []
        self._last_inference_start = None

    async def load_model(self) -> bool:
        """Load model asynchronously."""
        try:
            self.logger.info(f"Loading {self.config.type.value} model: {self.config.name}")
            start_time = time.time()

            # Implementation specific to backend
            success = await self._load_model_impl()

            load_time = time.time() - start_time
            self.logger.info(".2f"
            return success

        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.name}: {str(e)}")
            return False

    async def _load_model_impl(self) -> bool:
        """Backend-specific model loading implementation."""
        raise NotImplementedError("Subclasses must implement _load_model_impl")

    async def inference(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run model inference with performance monitoring."""
        try:
            self._last_inference_start = time.time()

            # Preprocessing
            processed_inputs = await self._preprocess_inputs(inputs)

            # Actual inference
            outputs = await self._inference_impl(processed_inputs, **kwargs)

            # Postprocessing
            final_outputs = await self._postprocess_outputs(outputs)

            # Update metrics
            self._update_metrics()

            return final_outputs

        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            self.metrics.error_rate += 1
            raise

    async def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs for model."""
        return inputs

    async def _inference_impl(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Backend-specific inference implementation."""
        raise NotImplementedError("Subclasses must implement _inference_impl")

    async def _postprocess_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess model outputs."""
        return outputs

    def _update_metrics(self):
        """Update performance metrics."""
        if self._last_inference_start:
            inference_time = time.time() - self._last_inference_start
            self._inference_times.append(inference_time)

            # Keep only recent measurements
            if len(self._inference_times) > 100:
                self._inference_times = self._inference_times[-100:]

            # Update metrics
            self.metrics.inference_time = inference_time
            self.metrics.latency_p50 = np.percentile(self._inference_times, 50)
            self.metrics.latency_p95 = np.percentile(self._inference_times, 95)
            self.metrics.latency_p99 = np.percentile(self._inference_times, 99)

    async def unload_model(self):
        """Unload model and free resources."""
        try:
            await self._unload_model_impl()
            self.model = None
            self.logger.info(f"Unloaded model: {self.config.name}")
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")

    async def _unload_model_impl(self):
        """Backend-specific model unloading."""
        pass

    def get_metrics(self) -> ModelMetrics:
        """Get current model metrics."""
        return self.metrics.copy()


class PyTorchModelWrapper(BaseModelWrapper):
    """PyTorch model wrapper for NVIDIA GPU optimization."""

    async def _load_model_impl(self) -> bool:
        """Load PyTorch model with NVIDIA optimizations."""
        try:
            model_path = Path(self.config.model_path)

            if not model_path.exists():
                self.logger.error(f"Model path does not exist: {model_path}")
                return False

            # Load tokenizer if available
            tokenizer_path = model_path / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

            # Load model with optimizations
            load_kwargs = {
                "torch_dtype": torch.float16 if self.config.precision == "fp16" else torch.float32,
                "device_map": "auto" if self.device.startswith("cuda") else None,
                "trust_remote_code": True
            }

            # Apply quantization if specified
            if self.config.quantization == "int8":
                load_kwargs["load_in_8bit"] = True
            elif self.config.quantization == "int4":
                load_kwargs["load_in_4bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **load_kwargs
            )

            # Move to specified device
            if self.device != "auto":
                self.model.to(self.device)

            # Enable optimizations
            if self.device.startswith("cuda"):
                self.model = torch.compile(self.model, mode="reduce-overhead")

            return True

        except Exception as e:
            self.logger.error(f"PyTorch model loading failed: {str(e)}")
            return False

    async def _inference_impl(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """PyTorch inference implementation."""
        with torch.no_grad():
            # Prepare inputs
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            voice_features = inputs.get("voice_features")

            if isinstance(input_ids, np.ndarray):
                input_ids = torch.from_numpy(input_ids).to(self.device)
            if isinstance(attention_mask, np.ndarray):
                attention_mask = torch.from_numpy(attention_mask).to(self.device)
            if voice_features is not None and isinstance(voice_features, np.ndarray):
                voice_features = torch.from_numpy(voice_features).to(self.device)

            # Model inputs
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Add voice features if available (for voice-enabled models)
            if voice_features is not None and hasattr(self.model, 'voice_adapter'):
                model_inputs["voice_features"] = voice_features

            # Generate response
            max_new_tokens = kwargs.get("max_new_tokens", 100)
            temperature = kwargs.get("temperature", 0.7)

            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "response": response_text,
                "input_length": len(input_ids[0]) if len(input_ids.shape) > 0 else len(input_ids),
                "output_length": len(outputs[0])
            }


class TritonModelWrapper(BaseModelWrapper):
    """Triton Inference Server model wrapper."""

    async def _load_model_impl(self) -> bool:
        """Initialize Triton client."""
        try:
            triton_url = self.config.triton_url or "localhost:8001"
            self.client = grpcclient.InferenceServerClient(url=triton_url)

            # Check if model is available
            model_ready = self.client.is_model_ready(self.config.name)
            if not model_ready:
                self.logger.error(f"Triton model not ready: {self.config.name}")
                return False

            # Get model metadata
            self.metadata = self.client.get_model_metadata(self.config.name)
            self.logger.info(f"Triton model loaded: {self.config.name}")
            return True

        except Exception as e:
            self.logger.error(f"Triton model loading failed: {str(e)}")
            return False

    async def _inference_impl(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Triton inference implementation."""
        try:
            # Prepare Triton inputs
            triton_inputs = []
            for input_name, input_data in inputs.items():
                # Find input metadata
                input_meta = None
                for inp in self.metadata.inputs:
                    if inp.name == input_name:
                        input_meta = inp
                        break

                if input_meta is None:
                    continue

                # Convert data type
                if isinstance(input_data, np.ndarray):
                    data = input_data
                else:
                    data = np.array(input_data)

                # Ensure correct dtype
                target_dtype = triton_to_np_dtype(input_meta.datatype)
                if data.dtype != target_dtype:
                    data = data.astype(target_dtype)

                triton_input = grpcclient.InferInput(input_name, data.shape, input_meta.datatype)
                triton_input.set_data_from_numpy(data)
                triton_inputs.append(triton_input)

            # Prepare outputs
            triton_outputs = []
            for output in self.metadata.outputs:
                triton_outputs.append(grpcclient.InferRequestedOutput(output.name))

            # Run inference
            response = self.client.infer(self.config.name, triton_inputs, outputs=triton_outputs)

            # Extract results
            results = {}
            for output in triton_outputs:
                results[output.name()] = response.as_numpy(output.name())

            return results

        except Exception as e:
            self.logger.error(f"Triton inference failed: {str(e)}")
            raise


class ModelManager:
    """Central model management following NVIDIA Maverick patterns."""

    def __init__(self):
        self.device_manager = NVIDIADeviceManager()
        self.loaded_models: Dict[str, BaseModelWrapper] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def register_model(self, config: ModelConfig):
        """Register a model configuration."""
        self.model_configs[config.name] = config
        self.logger.info(f"Registered model: {config.name}")

    async def load_model(self, model_name: str) -> bool:
        """Load a registered model."""
        if model_name not in self.model_configs:
            self.logger.error(f"Model not registered: {model_name}")
            return False

        if model_name in self.loaded_models:
            self.logger.warning(f"Model already loaded: {model_name}")
            return True

        config = self.model_configs[model_name]

        # Create appropriate wrapper based on backend
        if config.backend == ModelBackend.PYTORCH:
            wrapper = PyTorchModelWrapper(config, self.device_manager)
        elif config.backend == ModelBackend.TRITON:
            wrapper = TritonModelWrapper(config, self.device_manager)
        else:
            self.logger.error(f"Unsupported backend: {config.backend}")
            return False

        # Load the model
        success = await wrapper.load_model()
        if success:
            self.loaded_models[model_name] = wrapper
            return True
        else:
            return False

    async def unload_model(self, model_name: str):
        """Unload a model."""
        if model_name in self.loaded_models:
            await self.loaded_models[model_name].unload_model()
            del self.loaded_models[model_name]
            self.logger.info(f"Unloaded model: {model_name}")

    async def inference(self, model_name: str, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run inference on a loaded model."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model not loaded: {model_name}")

        model = self.loaded_models[model_name]
        return await model.inference(inputs, **kwargs)

    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a loaded model."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name].get_metrics()
        return None

    def list_loaded_models(self) -> List[str]:
        """List currently loaded models."""
        return list(self.loaded_models.keys())

    def list_registered_models(self) -> List[str]:
        """List registered model configurations."""
        return list(self.model_configs.keys())

    async def optimize_for_voice(self, model_name: str):
        """Apply voice-specific optimizations to a model."""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model not loaded: {model_name}")

        model = self.loaded_models[model_name]

        # Apply voice-specific optimizations
        if hasattr(model, 'enable_voice_mode'):
            await model.enable_voice_mode()

        self.logger.info(f"Applied voice optimizations to model: {model_name}")

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        return {
            "devices": self.device_manager.devices,
            "optimal_device": "cuda:0" if torch.cuda.is_available() else "cpu"
        }


# Global model manager instance
model_manager = ModelManager()


async def initialize_models(configs: List[ModelConfig]):
    """Initialize multiple models asynchronously."""
    tasks = []
    for config in configs:
        model_manager.register_model(config)
        tasks.append(model_manager.load_model(config.name))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if r is True)
    error_count = sum(1 for r in results if isinstance(r, Exception))

    logging.info(f"Model initialization complete: {success_count} success, {error_count} errors")

    return success_count, error_count


def create_voice_config(name: str, model_path: str, backend: ModelBackend = ModelBackend.PYTORCH) -> ModelConfig:
    """Create a voice-enabled model configuration."""
    return ModelConfig(
        name=name,
        type=ModelType.LLM,
        backend=backend,
        model_path=model_path,
        device="auto",
        precision="fp16",
        max_batch_size=1,
        max_sequence_length=2048,
        voice_embedding_dim=768,
        enable_streaming=True,
        quantization="none",
        custom_config={
            "voice_enabled": True,
            "emotion_recognition": True,
            "speech_context": True
        }
    )


def create_speech_config(name: str, model_path: str) -> ModelConfig:
    """Create a speech recognition model configuration."""
    return ModelConfig(
        name=name,
        type=ModelType.SPEECH_RECOGNITION,
        backend=ModelBackend.TRITON,
        model_path=model_path,
        device="auto",
        max_batch_size=8,
        custom_config={
            "language": "en-US",
            "vad_enabled": True
        }
    )


def create_tts_config(name: str, model_path: str) -> ModelConfig:
    """Create a text-to-speech model configuration."""
    return ModelConfig(
        name=name,
        type=ModelType.VOICE_SYNTHESIS,
        backend=ModelBackend.TENSORRT,
        model_path=model_path,
        device="auto",
        max_batch_size=4,
        custom_config={
            "sample_rate": 22050,
            "voice_cloning": True
        }
    )
