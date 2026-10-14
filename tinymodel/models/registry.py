"""Model Registry for discovering, instantiating, and validating models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from tinymodel.models.base import ModelMetadata, ModelWrapper
from tinymodel.models.builtin import (
    create_mini_transformer_text_clf,
    create_mobilenet_v3_small,
    create_resnet18,
    create_simple_mlp,
    create_tiny_convnet,
)


class ModelRegistry:
    """Thread-safe registry for neural network architectures in TinyModel Lab."""

    def __init__(self) -> None:
        self._registry: dict[str, tuple[Callable[..., ModelWrapper], ModelMetadata | None]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register default supported architectures."""
        self.register("resnet18", create_resnet18)
        self.register("mobilenet_v3_small", create_mobilenet_v3_small)
        self.register("tiny_convnet", create_tiny_convnet)
        self.register("simple_mlp", create_simple_mlp)
        self.register("mini_transformer_text", create_mini_transformer_text_clf)

    def register(
        self,
        model_id: str,
        factory_fn: Callable[..., ModelWrapper],
        metadata: ModelMetadata | None = None,
    ) -> None:
        """Register a new model factory."""
        self._registry[model_id.lower()] = (factory_fn, metadata)

    def get(
        self,
        model_id: str,
        pretrained: bool = False,
        device: torch.device | str = "cpu",
        **kwargs,
    ) -> ModelWrapper:
        """Instantiate and return a registered model wrapped in ModelWrapper."""
        key = model_id.lower()
        if key not in self._registry:
            available = ", ".join(self.list_model_ids())
            raise KeyError(f"Model '{model_id}' not found in registry. Available: {available}")

        factory_fn, _ = self._registry[key]
        wrapper = factory_fn(pretrained=pretrained, device=device, **kwargs)
        return wrapper

    def list_model_ids(self) -> list[str]:
        """Return list of all registered model IDs."""
        return sorted(list(self._registry.keys()))

    def list_models(self) -> list[ModelMetadata]:
        """Return metadata for all registered models."""
        models: list[ModelMetadata] = []
        for model_id in self.list_model_ids():
            models.append(self.get_metadata(model_id))
        return models

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Retrieve metadata for a specific registered model."""
        key = model_id.lower()
        if key not in self._registry:
            raise KeyError(f"Model '{model_id}' not found in registry.")

        factory_fn, metadata = self._registry[key]
        if metadata is not None:
            return metadata

        # Generate metadata by invoking factory with mock device
        wrapper = factory_fn(device="cpu")
        return wrapper.metadata

    def load_custom_module(
        self,
        module: nn.Module,
        metadata: ModelMetadata,
        weights_path: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> ModelWrapper:
        """Safely wrap an existing nn.Module instance with metadata and optional weights loading.

        Enforces safe state_dict deserialization (weights_only=True) to prevent
        arbitrary Python code execution from untrusted pickle payloads.
        """
        wrapper = ModelWrapper(module=module, metadata=metadata, device=device)
        if weights_path is not None:
            wrapper.load_weights(weights_path)
        return wrapper


# Global singleton instance
default_registry = ModelRegistry()
