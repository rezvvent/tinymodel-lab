"""Benchmark report and comparison data structures."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field
from rich.table import Table

from tinymodel.benchmarking.hardware import HardwareInfo
from tinymodel.utils.formatting import format_latency, format_params


class BenchmarkReport(BaseModel):
    """Complete, reproducible benchmark report for a single model execution."""

    model_id: str
    model_type: str  # "pytorch" | "onnx"
    hardware: HardwareInfo
    device: str
    batch_size: int
    input_shape: list[int]
    num_warmup_runs: int
    num_benchmark_runs: int

    # Structural metrics
    total_parameters: int
    trainable_parameters: int
    logical_sparsity: float
    model_size_mb: float
    load_time_ms: float

    # Latency metrics (in milliseconds)
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float

    # Throughput and memory
    throughput_samples_per_sec: float
    peak_memory_mb: float

    # Accuracy/quality metrics (optional)
    accuracy: float | None = None
    top5_accuracy: float | None = None
    f1_macro: float | None = None
    f1_weighted: float | None = None

    # Raw latencies for statistical verification
    raw_latencies_ms: list[float] = Field(default_factory=list)

    def to_rich_table(self) -> Table:
        """Create a formatted Rich Table summarizing benchmark metrics."""
        table = Table(title=f"Benchmark Report: {self.model_id} ({self.model_type.upper()})")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold green")

        table.add_row("Device", self.device)
        table.add_row("Batch Size", str(self.batch_size))
        table.add_row("Total Parameters", format_params(self.total_parameters))
        table.add_row("Logical Sparsity", f"{self.logical_sparsity * 100:.2f}%")
        table.add_row("Artifact Size", f"{self.model_size_mb:.2f} MB")
        table.add_row("Latency p50", format_latency(self.latency_p50_ms))
        table.add_row("Latency p95", format_latency(self.latency_p95_ms))
        table.add_row("Latency Mean ± Std", f"{self.latency_mean_ms:.2f} ± {self.latency_std_ms:.2f} ms")
        table.add_row("Throughput", f"{self.throughput_samples_per_sec:.1f} samples/s")
        table.add_row("Peak Memory", f"{self.peak_memory_mb:.2f} MB")

        if self.accuracy is not None:
            table.add_row("Accuracy", f"{self.accuracy * 100:.2f}%")
        if self.f1_macro is not None:
            table.add_row("F1 Macro", f"{self.f1_macro * 100:.2f}%")

        return table

    def save_json(self, path: str | Path) -> Path:
        """Save report to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return target_path

    @classmethod
    def load_json(cls, path: str | Path) -> BenchmarkReport:
        """Load report from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class BenchmarkComparison(BaseModel):
    """Fair, validated comparison between baseline and optimized models."""

    baseline_model_id: str
    optimized_model_id: str
    batch_size: int
    device: str

    # Compression ratios
    compression_ratio: float  # baseline_size / optimized_size
    size_reduction_percent: float  # (baseline - opt) / baseline * 100
    param_reduction_percent: float

    # Speedup and throughput
    speedup_ratio: float  # baseline_p50 / opt_p50
    throughput_gain_ratio: float  # opt_throughput / baseline_throughput

    # Quality impact
    accuracy_drop: float | None = None
    f1_macro_drop: float | None = None

    # Memory impact
    peak_memory_ratio: float

    # Discrepancies if run conditions differ
    fair_comparison: bool = True
    discrepancy_warnings: list[str] = Field(default_factory=list)

    @classmethod
    def compare(
        cls,
        baseline: BenchmarkReport,
        optimized: BenchmarkReport,
    ) -> BenchmarkComparison:
        """Compare two benchmark reports, validating identical hardware and conditions."""
        warnings: list[str] = []

        # Validate conditions
        is_comparable, hw_discrepancies = baseline.hardware.is_comparable_with(optimized.hardware)
        if not is_comparable:
            warnings.extend(hw_discrepancies)

        if baseline.batch_size != optimized.batch_size:
            warnings.append(
                f"Batch size mismatch: baseline={baseline.batch_size} vs optimized={optimized.batch_size}"
            )
        if baseline.input_shape != optimized.input_shape:
            warnings.append(
                f"Input shape mismatch: baseline={baseline.input_shape} vs optimized={optimized.input_shape}"
            )
        def normalize_dev(d: str) -> str:
            d_lower = d.lower()
            if "cuda" in d_lower:
                return "cuda"
            if "mps" in d_lower:
                return "mps"
            if "cpu" in d_lower:
                return "cpu"
            return d_lower

        if normalize_dev(baseline.device) != normalize_dev(optimized.device):
            warnings.append(
                f"Device mismatch: baseline={baseline.device} vs optimized={optimized.device}"
            )

        fair = len(warnings) == 0

        # Compute metrics
        comp_ratio = (
            round(baseline.model_size_mb / optimized.model_size_mb, 2)
            if optimized.model_size_mb > 0
            else 1.0
        )
        size_red_pct = round(
            ((baseline.model_size_mb - optimized.model_size_mb) / baseline.model_size_mb) * 100, 2
        ) if baseline.model_size_mb > 0 else 0.0

        param_red_pct = round(
            ((baseline.total_parameters - optimized.total_parameters) / baseline.total_parameters) * 100,
            2,
        ) if baseline.total_parameters > 0 else 0.0

        speedup = (
            round(baseline.latency_p50_ms / optimized.latency_p50_ms, 2)
            if optimized.latency_p50_ms > 0
            else 1.0
        )

        tp_gain = (
            round(optimized.throughput_samples_per_sec / baseline.throughput_samples_per_sec, 2)
            if baseline.throughput_samples_per_sec > 0
            else 1.0
        )

        acc_drop = None
        if baseline.accuracy is not None and optimized.accuracy is not None:
            acc_drop = round(baseline.accuracy - optimized.accuracy, 4)

        f1_drop = None
        if baseline.f1_macro is not None and optimized.f1_macro is not None:
            f1_drop = round(baseline.f1_macro - optimized.f1_macro, 4)

        mem_ratio = (
            round(baseline.peak_memory_mb / optimized.peak_memory_mb, 2)
            if optimized.peak_memory_mb > 0
            else 1.0
        )

        return cls(
            baseline_model_id=baseline.model_id,
            optimized_model_id=optimized.model_id,
            batch_size=baseline.batch_size,
            device=baseline.device,
            compression_ratio=comp_ratio,
            size_reduction_percent=size_red_pct,
            param_reduction_percent=param_red_pct,
            speedup_ratio=speedup,
            throughput_gain_ratio=tp_gain,
            accuracy_drop=acc_drop,
            f1_macro_drop=f1_drop,
            peak_memory_ratio=mem_ratio,
            fair_comparison=fair,
            discrepancy_warnings=warnings,
        )

    def to_rich_table(self) -> Table:
        """Create a formatted Rich Table summarizing the comparison."""
        title = f"Comparison: {self.baseline_model_id} (Base) vs {self.optimized_model_id} (Opt)"
        if not self.fair_comparison:
            title += " [WARNING: Conditions Differ]"

        table = Table(title=title)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Result", style="bold yellow")

        table.add_row("Fair Comparison", "✅ Yes" if self.fair_comparison else "⚠️ Conditions Differ")
        table.add_row("Compression Ratio", f"{self.compression_ratio:.2f}x")
        table.add_row("Size Reduction", f"{self.size_reduction_percent:.1f}%")
        table.add_row("Parameter Reduction", f"{self.param_reduction_percent:.1f}%")
        table.add_row("Speedup (p50 Latency)", f"{self.speedup_ratio:.2f}x")
        table.add_row("Throughput Gain", f"{self.throughput_gain_ratio:.2f}x")
        table.add_row("Memory Ratio", f"{self.peak_memory_ratio:.2f}x")

        if self.accuracy_drop is not None:
            color = "green" if self.accuracy_drop <= 0.01 else "red"
            table.add_row("Accuracy Drop", f"[{color}]{self.accuracy_drop * 100:.2f}%[/{color}]")

        if self.discrepancy_warnings:
            table.add_row("Warnings", "\n".join(self.discrepancy_warnings))

        return table
