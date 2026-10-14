"""Built-in supported neural network architectures for TinyModel Lab."""

from __future__ import annotations

import torch
import torch.nn as nn

from tinymodel.models.base import ModelMetadata, ModelWrapper, OptimizationType, TaskType


class TinyConvNet(nn.Module):
    """A clean, modular 3-layer Convolutional Neural Network.

    Designed for fast, reproducible testing of pruning, quantization,
    and knowledge distillation without heavy weights downloading.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for fast testing of dynamic quantization and linear pruning."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class MiniTransformerTextClassifier(nn.Module):
    """Compact Transformer model for text sequence classification."""

    def __init__(
        self,
        vocab_size: int = 2000,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        # Global average pooling over token sequence
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def create_resnet18(pretrained: bool = False, device: str = "cpu") -> ModelWrapper:
    """Instantiate torchvision ResNet18 wrapper."""
    import torchvision.models as models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    module = models.resnet18(weights=weights)

    metadata = ModelMetadata(
        model_id="resnet18",
        name="ResNet-18",
        task=TaskType.IMAGE_CLASSIFICATION,
        input_shape=(3, 224, 224),
        num_classes=1000,
        supported_optimizations=[
            OptimizationType.DYNAMIC_QUANTIZATION,
            OptimizationType.STATIC_QUANTIZATION,
            OptimizationType.ONNX_DYNAMIC_QUANTIZATION,
            OptimizationType.ONNX_STATIC_QUANTIZATION,
            OptimizationType.GLOBAL_L1_PRUNING,
            OptimizationType.LAYERWISE_L1_PRUNING,
            OptimizationType.STRUCTURED_CHANNEL_PRUNING,
            OptimizationType.KNOWLEDGE_DISTILLATION,
            OptimizationType.ONNX_EXPORT,
        ],
        description="Standard 18-layer residual network for computer vision.",
        tags=["vision", "cnn", "classification", "torchvision"],
    )
    return ModelWrapper(module, metadata, device=device)


def create_mobilenet_v3_small(pretrained: bool = False, device: str = "cpu") -> ModelWrapper:
    """Instantiate torchvision MobileNetV3-Small wrapper."""
    import torchvision.models as models

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    module = models.mobilenet_v3_small(weights=weights)

    metadata = ModelMetadata(
        model_id="mobilenet_v3_small",
        name="MobileNetV3-Small",
        task=TaskType.IMAGE_CLASSIFICATION,
        input_shape=(3, 224, 224),
        num_classes=1000,
        supported_optimizations=[
            OptimizationType.DYNAMIC_QUANTIZATION,
            OptimizationType.STATIC_QUANTIZATION,
            OptimizationType.ONNX_DYNAMIC_QUANTIZATION,
            OptimizationType.ONNX_STATIC_QUANTIZATION,
            OptimizationType.GLOBAL_L1_PRUNING,
            OptimizationType.LAYERWISE_L1_PRUNING,
            OptimizationType.KNOWLEDGE_DISTILLATION,
            OptimizationType.ONNX_EXPORT,
        ],
        description="Efficient mobile architecture optimized for low-latency edge deployment.",
        tags=["vision", "mobile", "efficient", "torchvision"],
    )
    return ModelWrapper(module, metadata, device=device)


def create_tiny_convnet(
    num_classes: int = 10,
    pretrained: bool = False,
    device: str = "cpu",
    **kwargs,
) -> ModelWrapper:
    """Instantiate lightweight TinyConvNet wrapper."""
    module = TinyConvNet(in_channels=3, num_classes=num_classes)
    metadata = ModelMetadata(
        model_id="tiny_convnet",
        name="TinyConvNet",
        task=TaskType.IMAGE_CLASSIFICATION,
        input_shape=(3, 32, 32),
        num_classes=num_classes,
        supported_optimizations=[
            OptimizationType.DYNAMIC_QUANTIZATION,
            OptimizationType.STATIC_QUANTIZATION,
            OptimizationType.ONNX_DYNAMIC_QUANTIZATION,
            OptimizationType.ONNX_STATIC_QUANTIZATION,
            OptimizationType.GLOBAL_L1_PRUNING,
            OptimizationType.LAYERWISE_L1_PRUNING,
            OptimizationType.STRUCTURED_CHANNEL_PRUNING,
            OptimizationType.KNOWLEDGE_DISTILLATION,
            OptimizationType.ONNX_EXPORT,
        ],
        description="Lightweight 3-layer CNN designed for fast unit testing and experimentation.",
        tags=["vision", "test", "lightweight", "custom"],
    )
    return ModelWrapper(module, metadata, device=device)


def create_simple_mlp(
    input_dim: int = 128,
    num_classes: int = 10,
    pretrained: bool = False,
    device: str = "cpu",
    **kwargs,
) -> ModelWrapper:
    """Instantiate SimpleMLP wrapper."""
    module = SimpleMLP(input_dim=input_dim, num_classes=num_classes)
    metadata = ModelMetadata(
        model_id="simple_mlp",
        name="SimpleMLP",
        task=TaskType.GENERIC,
        input_shape=(input_dim,),
        num_classes=num_classes,
        supported_optimizations=[
            OptimizationType.DYNAMIC_QUANTIZATION,
            OptimizationType.ONNX_DYNAMIC_QUANTIZATION,
            OptimizationType.GLOBAL_L1_PRUNING,
            OptimizationType.LAYERWISE_L1_PRUNING,
            OptimizationType.ONNX_EXPORT,
        ],
        description="Feedforward Multi-Layer Perceptron for dynamic quantization testing.",
        tags=["tabular", "mlp", "linear"],
    )
    return ModelWrapper(module, metadata, device=device)


def create_mini_transformer_text_clf(
    pretrained: bool = False,
    device: str = "cpu",
    **kwargs,
) -> ModelWrapper:
    """Instantiate MiniTransformer text classifier wrapper."""
    module = MiniTransformerTextClassifier()
    metadata = ModelMetadata(
        model_id="mini_transformer_text",
        name="MiniTransformer-Text",
        task=TaskType.TEXT_CLASSIFICATION,
        input_shape=(128,),
        num_classes=2,
        supported_optimizations=[
            OptimizationType.DYNAMIC_QUANTIZATION,
            OptimizationType.ONNX_DYNAMIC_QUANTIZATION,
            OptimizationType.GLOBAL_L1_PRUNING,
            OptimizationType.ONNX_EXPORT,
        ],
        description="Self-attention transformer for binary text classification.",
        tags=["nlp", "transformer", "classification"],
    )
    return ModelWrapper(module, metadata, device=device)
