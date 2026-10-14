"""Unit tests for ModelRegistry and ModelWrapper."""

import torch
import torch.nn as nn

from tinymodel.models.base import ModelMetadata, OptimizationType, TaskType
from tinymodel.models.registry import ModelRegistry, default_registry


def test_default_registry_contains_expected_models():
    """Verify built-in architectures are registered."""
    model_ids = default_registry.list_model_ids()
    expected = [
        "mini_transformer_text",
        "mobilenet_v3_small",
        "resnet18",
        "simple_mlp",
        "tiny_convnet",
    ]
    for exp in expected:
        assert exp in model_ids


def test_resnet18_metadata():
    """Verify ResNet-18 metadata specifications."""
    meta = default_registry.get_metadata("resnet18")
    assert meta.model_id == "resnet18"
    assert meta.task == TaskType.IMAGE_CLASSIFICATION
    assert meta.input_shape == (3, 224, 224)
    assert meta.supports(OptimizationType.ONNX_EXPORT)
    assert meta.supports(OptimizationType.DYNAMIC_QUANTIZATION)
    assert meta.supports(OptimizationType.STRUCTURED_CHANNEL_PRUNING)


def test_model_wrapper_properties(tiny_convnet):
    """Verify parameter counting and device reporting."""
    assert tiny_convnet.total_parameters > 0
    assert tiny_convnet.trainable_parameters == tiny_convnet.total_parameters
    assert tiny_convnet.non_trainable_parameters == 0
    assert tiny_convnet.primary_dtype == torch.float32
    assert tiny_convnet.parameter_size_bytes > 0


def test_dummy_input_generation(tiny_convnet):
    """Verify dummy input matches input_shape."""
    dummy = tiny_convnet.get_dummy_input(batch_size=2)
    assert dummy.shape == (2, 3, 32, 32)
    assert dummy.dtype == torch.float32


def test_text_model_dummy_input():
    """Verify text model dummy input generates discrete token IDs."""
    text_model = default_registry.get("mini_transformer_text")
    dummy = text_model.get_dummy_input(batch_size=4)
    assert dummy.shape == (4, 128)
    assert dummy.dtype == torch.long


def test_safe_weights_saving_and_loading(tmp_path, tiny_convnet):
    """Verify state_dict saving and secure weights_only loading."""
    weights_path = tmp_path / "model_weights.pt"
    tiny_convnet.save_weights(weights_path)
    assert weights_path.exists()

    # Load into a new fresh instance
    new_model = default_registry.get("tiny_convnet")
    new_model.load_weights(weights_path, strict=True)

    # Validate output equivalence
    dummy = tiny_convnet.get_dummy_input(batch_size=1)
    out1 = tiny_convnet(dummy)
    out2 = new_model(dummy)
    assert torch.allclose(out1, out2)


def test_custom_module_safe_loader():
    """Verify loading custom user Module with strict state_dict checking."""
    custom_net = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    meta = ModelMetadata(
        model_id="custom_seq",
        name="CustomSeq",
        task=TaskType.GENERIC,
        input_shape=(10,),
    )
    reg = ModelRegistry()
    wrapper = reg.load_custom_module(custom_net, meta)
    assert wrapper.total_parameters == (10 * 5 + 5) + (5 * 2 + 2)
