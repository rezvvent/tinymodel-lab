"""Quickstart Example: Profiling, Exporting, and Benchmarking ResNet-18 with TinyModel Lab."""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from tinymodel.benchmarking import BenchmarkComparison, BenchmarkRunner
from tinymodel.exporters import ONNXExporter, ONNXModelWrapper
from tinymodel.models import default_registry
from tinymodel.profiling import ModelProfiler

console = Console()


def main() -> None:
    console.print(Panel.fit("[bold green]TinyModel Lab — ResNet18 Quickstart Demo[/bold green]"))

    # 1. Load model from registry
    console.print("\n[bold cyan]1. Loading ResNet-18 from ModelRegistry...[/bold cyan]")
    model_wrapper = default_registry.get("resnet18", pretrained=False)
    console.print(f"Loaded: [bold]{model_wrapper.metadata.name}[/bold]")
    console.print(f"Parameters: {model_wrapper.total_parameters:,} ({model_wrapper.parameter_size_bytes / (1024 * 1024):.2f} MB)")

    # 2. Profile the baseline model
    console.print("\n[bold cyan]2. Profiling model architecture...[/bold cyan]")
    profiler = ModelProfiler()
    profile = profiler.profile(model_wrapper, batch_size=1)
    console.print(f"Total MACs:  {profile.estimated_macs:,} ({profile.estimated_macs / 1e9:.2f} G)")
    console.print(f"Total FLOPs: {profile.estimated_flops:,} ({profile.estimated_flops / 1e9:.2f} G)")
    console.print(f"Sparsity:    {profile.global_sparsity * 100:.2f}%")

    # 3. Export to ONNX
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "resnet18.onnx"

    console.print(f"\n[bold cyan]3. Exporting to ONNX at {onnx_path}...[/bold cyan]")
    exporter = ONNXExporter(default_opset=18)
    export_result = exporter.export(
        model_wrapper=model_wrapper,
        output_path=onnx_path,
        dynamic_batch=True,
        verify=True,
    )
    console.print(f"Export verified: [bold]{export_result.verified}[/bold]")
    console.print(f"ONNX Nodes:      {export_result.node_count}")
    console.print(f"Max Abs Error:   {export_result.max_absolute_error_vs_torch:.2e}")

    # 4. Benchmark baseline PyTorch vs ONNX Runtime
    console.print("\n[bold cyan]4. Benchmarking PyTorch vs ONNX Runtime (CPU)...[/bold cyan]")
    runner = BenchmarkRunner()

    console.print("Running PyTorch benchmark (10 warmup, 30 runs)...")
    pt_report = runner.benchmark_pytorch(
        model_wrapper=model_wrapper,
        batch_size=1,
        num_warmup_runs=10,
        num_benchmark_runs=30,
        device="cpu",
    )

    console.print("Running ONNX Runtime benchmark (10 warmup, 30 runs)...")
    onnx_wrapper = ONNXModelWrapper(onnx_path)
    onnx_report = runner.benchmark_onnx(
        onnx_wrapper=onnx_wrapper,
        input_shape=(3, 224, 224),
        model_id="resnet18_onnx",
        batch_size=1,
        num_warmup_runs=10,
        num_benchmark_runs=30,
    )

    # 5. Display comparison
    console.print("\n[bold cyan]5. Fair Comparison Report:[/bold cyan]")
    comparison = BenchmarkComparison.compare(pt_report, onnx_report)
    console.print(comparison.to_rich_table())

    console.print(f"\nPyTorch Latency (p50): {pt_report.latency_p50_ms:.2f} ms")
    console.print(f"ONNX Latency (p50):    {onnx_report.latency_p50_ms:.2f} ms")
    console.print(f"Measured Speedup:      [bold green]{comparison.speedup_ratio:.2f}x[/bold green]")


if __name__ == "__main__":
    main()
