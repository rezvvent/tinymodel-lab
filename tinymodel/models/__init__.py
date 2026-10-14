"""Model abstractions, wrappers, and registry."""

from tinymodel.models.base import ModelMetadata, ModelWrapper, OptimizationType, TaskType
from tinymodel.models.builtin import (
    MiniTransformerTextClassifier,
    SimpleMLP,
    TinyConvNet,
    create_mini_transformer_text_clf,
    create_mobilenet_v3_small,
    create_resnet18,
    create_simple_mlp,
    create_tiny_convnet,
)
from tinymodel.models.registry import ModelRegistry, default_registry

__all__ = [
    "ModelMetadata",
    "ModelWrapper",
    "TaskType",
    "OptimizationType",
    "ModelRegistry",
    "default_registry",
    "TinyConvNet",
    "SimpleMLP",
    "MiniTransformerTextClassifier",
    "create_resnet18",
    "create_mobilenet_v3_small",
    "create_tiny_convnet",
    "create_simple_mlp",
    "create_mini_transformer_text_clf",
]
