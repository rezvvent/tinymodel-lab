"""Pytest fixtures for TinyModel Lab test suite."""

import pytest

from tinymodel.datasets.synthetic import create_synthetic_dataloader
from tinymodel.models.registry import default_registry


@pytest.fixture
def tiny_convnet():
    """Return an instantiated TinyConvNet wrapper."""
    return default_registry.get("tiny_convnet")


@pytest.fixture
def simple_mlp():
    """Return an instantiated SimpleMLP wrapper."""
    return default_registry.get("simple_mlp")


@pytest.fixture
def synthetic_dataloader():
    """Return a deterministic DataLoader for vision testing."""
    return create_synthetic_dataloader(
        num_samples=20,
        batch_size=4,
        input_shape=(3, 32, 32),
        num_classes=10,
        seed=42,
    )
