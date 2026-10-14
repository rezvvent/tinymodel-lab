"""Unit tests for ONNX export and ONNX Runtime inference."""

import numpy as np
import torch

from tinymodel.exporters.onnx import ONNXExporter, ONNXModelWrapper


def test_onnx_export_tiny_convnet(tmp_path, tiny_convnet):
    """Verify exporting TinyConvNet to ONNX and numerical parity."""
    output_path = tmp_path / "tiny_convnet.onnx"
    exporter = ONNXExporter(default_opset=18)

    result = exporter.export(
        model_wrapper=tiny_convnet,
        output_path=output_path,
        verify=True,
    )

    assert output_path.exists()
    assert result.verified is True
    assert result.file_size_bytes > 0
    assert result.node_count > 0
    # Numerical error between PyTorch and ONNX Runtime must be minimal (< 1e-4)
    assert result.max_absolute_error_vs_torch < 1e-4


def test_onnx_model_wrapper_inference(tmp_path, simple_mlp):
    """Verify ONNXModelWrapper runs inference with both torch.Tensor and numpy.ndarray."""
    output_path = tmp_path / "simple_mlp.onnx"
    exporter = ONNXExporter(default_opset=18)
    exporter.export(simple_mlp, output_path, verify=True)

    ort_wrapper = ONNXModelWrapper(output_path)
    assert "CPUExecutionProvider" in ort_wrapper.active_providers

    # Forward with torch.Tensor
    x_torch = torch.randn(2, 128)
    out_torch = ort_wrapper.forward(x_torch)
    assert len(out_torch) == 1
    assert out_torch[0].shape == (2, 10)

    # Forward with np.ndarray
    x_numpy = np.random.randn(3, 128).astype(np.float32)
    out_numpy = ort_wrapper.forward(x_numpy)
    assert len(out_numpy) == 1
    assert out_numpy[0].shape == (3, 10)


def test_onnx_dynamic_batch(tmp_path, simple_mlp):
    """Verify dynamic batch support in exported ONNX model."""
    output_path = tmp_path / "dynamic_mlp.onnx"
    exporter = ONNXExporter(default_opset=18)
    exporter.export(simple_mlp, output_path, dynamic_batch=True)

    ort_wrapper = ONNXModelWrapper(output_path)
    for batch_size in [1, 4, 16]:
        inp = np.random.randn(batch_size, 128).astype(np.float32)
        out = ort_wrapper.forward(inp)
        assert out[0].shape == (batch_size, 10)
