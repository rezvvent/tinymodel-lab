"""Unit tests for ModelProfiler."""

import torch
import torch.nn as nn

from tinymodel.models.base import ModelMetadata, ModelWrapper, TaskType
from tinymodel.profiling.profiler import ModelProfiler


def test_profiler_parameter_counts(tiny_convnet):
    """Verify profiler accurately counts parameters."""
    profiler = ModelProfiler()
    report = profiler.profile(tiny_convnet)

    expected_params = sum(p.numel() for p in tiny_convnet.module.parameters())
    assert report.total_parameters == expected_params
    assert report.trainable_parameters == expected_params
    assert report.non_trainable_parameters == 0
    assert report.parameter_size_mb > 0
    assert report.total_size_mb >= report.parameter_size_mb


def test_profiler_sparsity_calculation():
    """Verify logical sparsity detection when weights are zeroed."""
    # Create simple linear network with 100 parameters
    linear = nn.Linear(10, 10, bias=False)  # 100 weights
    with torch.no_grad():
        # Set half the weights to zero
        linear.weight[:5, :] = 0.0

    meta = ModelMetadata(
        model_id="sparse_linear",
        name="SparseLinear",
        task=TaskType.GENERIC,
        input_shape=(10,),
    )
    wrapper = ModelWrapper(linear, meta)

    profiler = ModelProfiler()
    report = profiler.profile(wrapper)

    assert report.total_elements == 100
    assert report.total_zero_parameters == 50
    assert report.global_sparsity == 0.50


def test_profiler_macs_and_flops(simple_mlp):
    """Verify MACs and FLOPs calculation for MLP."""
    profiler = ModelProfiler()
    report = profiler.profile(simple_mlp, batch_size=1)

    # SimpleMLP:
    # fc1: 128 * 64 = 8192 MACs
    # fc2: 64 * 32 = 2048 MACs
    # fc3: 32 * 10 = 320 MACs
    # Total MACs = 10560
    assert report.estimated_macs == 10560
    assert report.estimated_flops == 2 * report.estimated_macs
    assert len(report.layers) == 3


def test_profiler_rich_table_rendering(tiny_convnet):
    """Verify Rich table generation from profiling report."""
    profiler = ModelProfiler()
    report = profiler.profile(tiny_convnet)
    table = report.to_rich_table()
    assert table.title == "Model Profile: tiny_convnet"
    assert len(table.rows) > 0
