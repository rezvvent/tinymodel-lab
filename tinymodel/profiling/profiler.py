"""Model profiling engine for parameter counts, sparsity, memory, and FLOPs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from rich.table import Table

from tinymodel.models.base import ModelWrapper
from tinymodel.utils.formatting import format_params


class LayerProfile(BaseModel):
    """Detailed profile of a single layer."""

    name: str
    class_name: str
    weight_shape: list[int] | None = None
    parameters_count: int = 0
    trainable: bool = True
    zeros_count: int = 0
    sparsity: float = 0.0
    dtype: str = "float32"
    macs: int = 0
    output_shape: list[int] | None = None


class ModelProfileReport(BaseModel):
    """Comprehensive profiling report for a neural network."""

    model_id: str
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    parameter_size_mb: float
    buffer_size_mb: float
    total_size_mb: float

    # Sparsity metrics
    total_elements: int
    total_zero_parameters: int
    global_sparsity: float

    # Computational complexity
    estimated_macs: int
    estimated_flops: int

    # Detailed layers
    layers: list[LayerProfile] = Field(default_factory=list)

    def to_rich_table(self) -> Table:
        """Generate a formatted Rich Table summary for terminal display."""
        table = Table(title=f"Model Profile: {self.model_id}")
        table.add_column("Layer Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Shape", style="green")
        table.add_column("Params", justify="right")
        table.add_column("Sparsity", justify="right")
        table.add_column("MACs", justify="right")

        for layer in self.layers:
            shape_str = str(layer.weight_shape) if layer.weight_shape else "-"
            table.add_row(
                layer.name,
                layer.class_name,
                shape_str,
                format_params(layer.parameters_count),
                f"{layer.sparsity * 100:.1f}%",
                format_params(layer.macs) if layer.macs > 0 else "-",
            )

        return table


class ModelProfiler:
    """Calculates parameter counts, logical sparsity, memory footprint, and FLOPs."""

    def profile(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int = 1,
    ) -> ModelProfileReport:
        """Perform full profile of model wrapper."""
        module = model_wrapper.module
        total_params = model_wrapper.total_parameters
        trainable_params = model_wrapper.trainable_parameters
        non_trainable_params = model_wrapper.non_trainable_parameters

        param_bytes = model_wrapper.parameter_size_bytes
        buffer_bytes = model_wrapper.buffer_size_bytes
        total_bytes = model_wrapper.total_size_bytes

        total_elements = 0
        total_zeros = 0
        layers: list[LayerProfile] = []

        # Hook-based MACs and output shape tracking
        macs_dict: dict[str, int] = {}
        output_shapes_dict: dict[str, list[int]] = {}
        hooks = []

        def make_hook(name: str):
            def hook_fn(sub_module: nn.Module, inputs: Any, output: Any):
                nonlocal macs_dict, output_shapes_dict
                # Record output shape
                if isinstance(output, torch.Tensor):
                    output_shapes_dict[name] = list(output.shape)
                elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    output_shapes_dict[name] = list(output[0].shape)

                # Estimate MACs
                macs = 0
                if isinstance(sub_module, nn.Conv2d):
                    # MACs = Cin * Kh * Kw * Cout * Hout * Wout / groups
                    if isinstance(output, torch.Tensor):
                        batch_s, cout, hout, wout = output.shape
                        cin = sub_module.in_channels
                        kh, kw = sub_module.kernel_size
                        groups = sub_module.groups
                        macs = int((cin * kh * kw * cout * hout * wout * batch_s) / groups)
                elif isinstance(sub_module, nn.Linear):
                    # MACs = in_features * out_features * batch_size
                    in_f = sub_module.in_features
                    out_f = sub_module.out_features
                    if isinstance(inputs, tuple) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                        in_tensor = inputs[0]
                        batch_mult = int(np.prod(in_tensor.shape[:-1]))
                        macs = int(in_f * out_f * batch_mult)
                    else:
                        macs = int(in_f * out_f * batch_size)

                macs_dict[name] = macs

            return hook_fn

        # Register forward hooks on leaf modules
        for name, sub_mod in module.named_modules():
            # If leaf module with parameters
            if len(list(sub_mod.children())) == 0 and len(list(sub_mod.parameters())) > 0:
                hooks.append(sub_mod.register_forward_hook(make_hook(name)))

        # Run forward pass with dummy input to trigger hooks
        dummy_input = model_wrapper.get_dummy_input(batch_size=batch_size)
        was_training = module.training
        module.eval()
        with torch.inference_mode():
            try:
                module(dummy_input)
            except Exception:
                # If forward pass fails with single tensor, try kwargs or ignore hook errors
                pass
            finally:
                for h in hooks:
                    h.remove()
                if was_training:
                    module.train()

        # Detailed layer analysis
        for name, sub_mod in module.named_modules():
            sub_params = list(sub_mod.parameters(recurse=False))
            if not sub_params:
                continue

            layer_params_count = sum(p.numel() for p in sub_params)
            layer_zeros = 0
            weight_shape: list[int] | None = None
            primary_dtype = "float32"

            for p_name, param in sub_mod.named_parameters(recurse=False):
                total_elements += param.numel()
                zeros = int((param == 0).sum().item())
                total_zeros += zeros
                layer_zeros += zeros

                if p_name == "weight":
                    weight_shape = list(param.shape)
                    primary_dtype = str(param.dtype).replace("torch.", "")

            sparsity = float(layer_zeros / layer_params_count) if layer_params_count > 0 else 0.0

            layers.append(
                LayerProfile(
                    name=name if name else "root",
                    class_name=sub_mod.__class__.__name__,
                    weight_shape=weight_shape,
                    parameters_count=layer_params_count,
                    trainable=any(p.requires_grad for p in sub_params),
                    zeros_count=layer_zeros,
                    sparsity=sparsity,
                    dtype=primary_dtype,
                    macs=macs_dict.get(name, 0),
                    output_shape=output_shapes_dict.get(name, None),
                )
            )

        global_sparsity = float(total_zeros / total_elements) if total_elements > 0 else 0.0
        total_macs = sum(layer.macs for layer in layers)
        total_flops = 2 * total_macs  # Standard FLOPs convention (1 MAC = 2 FLOPs)

        return ModelProfileReport(
            model_id=model_wrapper.metadata.model_id,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            non_trainable_parameters=non_trainable_params,
            parameter_size_mb=round(param_bytes / (1024 * 1024), 3),
            buffer_size_mb=round(buffer_bytes / (1024 * 1024), 3),
            total_size_mb=round(total_bytes / (1024 * 1024), 3),
            total_elements=total_elements,
            total_zero_parameters=total_zeros,
            global_sparsity=round(global_sparsity, 5),
            estimated_macs=total_macs,
            estimated_flops=total_flops,
            layers=layers,
        )
