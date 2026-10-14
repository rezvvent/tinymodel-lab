"""ONNX export and ONNX Runtime inference engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from pydantic import BaseModel

from tinymodel.models.base import ModelWrapper


class ONNXExportResult(BaseModel):
    """Result and metadata of an ONNX export."""

    output_path: str
    file_size_bytes: int
    file_size_mb: float
    opset_version: int
    dynamic_batch: bool
    input_names: list[str]
    output_names: list[str]
    node_count: int
    max_absolute_error_vs_torch: float
    verified: bool


class ONNXModelWrapper:
    """Encapsulates an ONNX Runtime InferenceSession for uniform evaluation and benchmarking."""

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None and num_threads > 0:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads

        # Determine available execution providers
        available_providers = ort.get_available_providers()
        selected_providers: list[str] = []
        if providers:
            for p in providers:
                if p in available_providers:
                    selected_providers.append(p)
        if not selected_providers:
            # Fallback to standard CPU
            selected_providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=selected_providers,
        )
        self.active_providers = self.session.get_providers()
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    @property
    def file_size_bytes(self) -> int:
        return self.model_path.stat().st_size

    @property
    def file_size_mb(self) -> float:
        return round(self.file_size_bytes / (1024 * 1024), 3)

    def forward(self, input_data: torch.Tensor | np.ndarray | dict[str, Any]) -> list[np.ndarray]:
        """Run inference using ONNX Runtime."""
        if isinstance(input_data, torch.Tensor):
            numpy_input = input_data.detach().cpu().numpy()
            feed_dict = {self.input_names[0]: numpy_input}
        elif isinstance(input_data, np.ndarray):
            feed_dict = {self.input_names[0]: input_data}
        elif isinstance(input_data, dict):
            feed_dict = {}
            for k, v in input_data.items():
                feed_dict[k] = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        else:
            raise TypeError(f"Unsupported input type for ONNX inference: {type(input_data)}")

        return self.session.run(self.output_names, feed_dict)

    def __call__(self, *args: Any, **kwargs: Any) -> list[np.ndarray]:
        return self.forward(*args, **kwargs)


class ONNXExporter:
    """Exports PyTorch models to ONNX format with graph validation and numerical verification."""

    def __init__(self, default_opset: int = 18) -> None:
        self.default_opset = default_opset

    def export(
        self,
        model_wrapper: ModelWrapper,
        output_path: str | Path,
        opset_version: int | None = None,
        dynamic_batch: bool = True,
        verify: bool = True,
        batch_size: int = 1,
        dynamo: bool = False,
    ) -> ONNXExportResult:
        """Export model wrapper to ONNX and run verification."""
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        opset = opset_version or self.default_opset

        module = model_wrapper.module
        was_training = module.training
        module.eval()

        dummy_input = model_wrapper.get_dummy_input(batch_size=batch_size, device="cpu")
        module_cpu = module.to("cpu")

        input_names = model_wrapper.metadata.input_names or ["input"]
        output_names = model_wrapper.metadata.output_names or ["output"]

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            }

        # Run reference PyTorch inference for numerical validation
        with torch.inference_mode():
            torch_output = module_cpu(dummy_input)

        # PyTorch to ONNX export
        torch.onnx.export(
            module_cpu,
            dummy_input,
            str(target_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=dynamo,
        )

        if was_training:
            module.train()

        # Check model structure with ONNX library
        onnx_model = onnx.load(str(target_path))
        if verify:
            onnx.checker.check_model(onnx_model)

        node_count = len(onnx_model.graph.node)
        file_size = target_path.stat().st_size

        # Numerical verification with ONNX Runtime
        ort_wrapper = ONNXModelWrapper(target_path)
        ort_outputs = ort_wrapper.forward(dummy_input)

        if isinstance(torch_output, torch.Tensor):
            ref_np = torch_output.detach().cpu().numpy()
        elif isinstance(torch_output, (tuple, list)):
            ref_np = torch_output[0].detach().cpu().numpy()
        else:
            ref_np = np.array(torch_output)

        max_abs_err = float(np.max(np.abs(ref_np - ort_outputs[0])))

        return ONNXExportResult(
            output_path=str(target_path),
            file_size_bytes=file_size,
            file_size_mb=round(file_size / (1024 * 1024), 3),
            opset_version=opset,
            dynamic_batch=dynamic_batch,
            input_names=input_names,
            output_names=output_names,
            node_count=node_count,
            max_absolute_error_vs_torch=max_abs_err,
            verified=verify and (max_abs_err < 1e-4),
        )
