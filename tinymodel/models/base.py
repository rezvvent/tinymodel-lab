"""Base interfaces and metadata models for neural networks in TinyModel Lab."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Supported machine learning tasks."""

    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_CLASSIFICATION = "text_classification"
    GENERIC = "generic"


class OptimizationType(StrEnum):
    """Supported model optimization methods."""

    DYNAMIC_QUANTIZATION = "dynamic_quantization"
    STATIC_QUANTIZATION = "static_quantization"
    ONNX_DYNAMIC_QUANTIZATION = "onnx_dynamic_quantization"
    ONNX_STATIC_QUANTIZATION = "onnx_static_quantization"
    GLOBAL_L1_PRUNING = "global_l1_pruning"
    LAYERWISE_L1_PRUNING = "layerwise_l1_pruning"
    STRUCTURED_CHANNEL_PRUNING = "structured_channel_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    ONNX_EXPORT = "onnx_export"


class ModelMetadata(BaseModel):
    """Metadata describing a registered model architecture and capabilities."""

    model_id: str
    name: str
    task: TaskType
    input_shape: tuple[int, ...]
    input_names: list[str] = Field(default_factory=lambda: ["input"])
    output_names: list[str] = Field(default_factory=lambda: ["output"])
    num_classes: int = 1000
    supported_optimizations: list[OptimizationType] = Field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)

    def supports(self, opt: OptimizationType | str) -> bool:
        """Check whether the model supports a specific optimization technique."""
        if isinstance(opt, str):
            try:
                opt = OptimizationType(opt)
            except ValueError:
                return False
        return opt in self.supported_optimizations


class ModelWrapper:
    """Encapsulates a PyTorch nn.Module with metadata, device management, and inspection utilities."""

    def __init__(
        self,
        module: nn.Module,
        metadata: ModelMetadata,
        device: torch.device | str = "cpu",
    ) -> None:
        self.metadata = metadata
        self.device = torch.device(device) if isinstance(device, str) else device
        self.module = module.to(self.device)
        self.module.eval()

    @property
    def total_parameters(self) -> int:
        """Total number of parameters in the model."""
        return sum(p.numel() for p in self.module.parameters())

    @property
    def trainable_parameters(self) -> int:
        """Number of trainable parameters in the model."""
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    @property
    def non_trainable_parameters(self) -> int:
        """Number of non-trainable (frozen) parameters."""
        return sum(p.numel() for p in self.module.parameters() if not p.requires_grad)

    @property
    def primary_dtype(self) -> torch.dtype:
        """Get the primary dtype of the model weights."""
        for p in self.module.parameters():
            return p.dtype
        for b in self.module.buffers():
            return b.dtype
        return torch.float32

    @property
    def parameter_size_bytes(self) -> int:
        """Estimated size in bytes of all parameters."""
        return sum(p.numel() * p.element_size() for p in self.module.parameters())

    @property
    def buffer_size_bytes(self) -> int:
        """Estimated size in bytes of all buffers (e.g. running mean/var in BatchNorm)."""
        return sum(b.numel() * b.element_size() for b in self.module.buffers())

    @property
    def total_size_bytes(self) -> int:
        """Total in-memory tensor size (parameters + buffers)."""
        return self.parameter_size_bytes + self.buffer_size_bytes

    def to(self, device: torch.device | str) -> ModelWrapper:
        """Move the underlying module to the specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.module.to(self.device)
        return self

    def get_dummy_input(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Generate a deterministic dummy input tensor matching the expected input shape."""
        target_device = (
            (torch.device(device) if isinstance(device, str) else device)
            if device is not None
            else self.device
        )
        shape = (batch_size, *self.metadata.input_shape)

        if self.metadata.task == TaskType.TEXT_CLASSIFICATION:
            # Synthetic token IDs
            return torch.randint(1, 1000, shape, dtype=torch.long, device=target_device)

        # Standard normalized continuous inputs (images/features)
        torch.manual_seed(42)
        return torch.randn(shape, dtype=torch.float32, device=target_device)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the underlying module."""
        return self.module(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def save_weights(self, path: str | Path) -> Path:
        """Safely save state_dict to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), target_path)
        return target_path

    def load_weights(self, path: str | Path, strict: bool = True) -> None:
        """Safely load state_dict using weights_only=True to prevent arbitrary code execution."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Weights file not found at: {target_path}")

        # weights_only=True is required for secure unpickling
        state_dict = torch.load(target_path, map_location=self.device, weights_only=True)
        self.module.load_state_dict(state_dict, strict=strict)
