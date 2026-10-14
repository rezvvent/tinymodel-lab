"""CLI interface for TinyModel Lab using Typer and Rich."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tinymodel.benchmarking.hardware import HardwareInfo
from tinymodel.benchmarking.metrics import BenchmarkComparison, BenchmarkReport
from tinymodel.benchmarking.runner import BenchmarkRunner
from tinymodel.exporters.onnx import ONNXExporter, ONNXModelWrapper
from tinymodel.models.registry import default_registry
from tinymodel.profiling.profiler import ModelProfiler
from tinymodel.utils.formatting import format_bytes, format_params

app = typer.Typer(
    name="tinymodel",
    help="TinyModel Lab — ML Model Compression, Profiling & Benchmarking Platform",
    add_completion=False,
)
console = Console()


@app.command("list-models")
def list_models() -> None:
    """List all registered neural network models and their metadata."""
    models = default_registry.list_models()
    table = Table(title="TinyModel Lab — Registered Architectures")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Task", style="magenta")
    table.add_column("Input Shape", style="green")
    table.add_column("Tags", style="yellow")

    for m in models:
        table.add_row(
            m.model_id,
            m.name,
            m.task.value,
            str(m.input_shape),
            ", ".join(m.tags),
        )

    console.print(table)


@app.command("hardware")
def show_hardware() -> None:
    """Display current hardware, OS, and ML runtime environment metadata."""
    hw = HardwareInfo.capture()
    table = Table(title="Environment & Hardware Snapshot")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Operating System", f"{hw.os_name} {hw.os_release} ({hw.machine})")
    table.add_row("Python Version", hw.python_version)
    table.add_row("Physical CPU Cores", str(hw.cpu_count_physical))
    table.add_row("Logical CPU Cores", str(hw.cpu_count_logical))
    table.add_row("PyTorch Threads", str(hw.torch_num_threads))
    table.add_row("Total RAM", format_bytes(hw.ram_total_bytes))
    table.add_row("PyTorch Version", hw.torch_version)
    table.add_row(
        "CUDA Available",
        f"✅ {hw.cuda_device_name}" if hw.cuda_available else "❌ Unavailable",
    )
    table.add_row("Apple MPS Available", "✅ Available" if hw.mps_available else "❌ Unavailable")
    table.add_row("ONNX Version", hw.onnx_version or "Not installed")
    table.add_row("ONNX Runtime Version", hw.onnxruntime_version or "Not installed")

    console.print(table)


@app.command("profile")
def profile(
    model: str = typer.Option(..., "--model", "-m", help="Registered model ID (e.g. resnet18, tiny_convnet)"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size for profiling forward pass"),
    details: bool = typer.Option(False, "--details", "-d", help="Show full layer-by-layer breakdown table"),
    json_out: Path | None = typer.Option(None, "--json-out", "-o", help="Optional path to save profile JSON"),
) -> None:
    """Profile neural network architecture: parameters, sparsity, FLOPs, and memory."""
    try:
        model_wrapper = default_registry.get(model)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Profiling model:[/cyan] [bold]{model}[/bold] (batch_size={batch_size})...")
    profiler = ModelProfiler()
    report = profiler.profile(model_wrapper, batch_size=batch_size)

    # Summary Panel
    summary_text = (
        f"Total Parameters:    [bold]{format_params(report.total_parameters)}[/bold] ({report.total_parameters:,})\n"
        f"Trainable Params:    [bold]{format_params(report.trainable_parameters)}[/bold]\n"
        f"Logical Sparsity:    [bold]{report.global_sparsity * 100:.2f}%[/bold] ({report.total_zero_parameters:,} zeros)\n"
        f"Estimated MACs:      [bold]{format_params(report.estimated_macs)}[/bold]\n"
        f"Estimated FLOPs:     [bold]{format_params(report.estimated_flops)}[/bold]\n"
        f"Weights Memory Size: [bold]{report.parameter_size_mb:.2f} MB[/bold]\n"
        f"Total Memory Size:   [bold]{report.total_size_mb:.2f} MB[/bold]"
    )
    console.print(Panel(summary_text, title=f"Profile Summary: {model}", border_style="green"))

    if details:
        console.print(report.to_rich_table())

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        console.print(f"[green]Profile saved to:[/green] {json_out}")


@app.command("export-onnx")
def export_onnx(
    model: str = typer.Option(..., "--model", "-m", help="Registered model ID to export"),
    output: Path = typer.Option(..., "--output", "-o", help="Target .onnx file path"),
    opset: int = typer.Option(18, "--opset", help="ONNX operator set version"),
    dynamic_batch: bool = typer.Option(True, "--dynamic-batch/--fixed-batch", help="Enable dynamic batch axis"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Run ONNX checker and numerical parity verification"),
) -> None:
    """Export a registered PyTorch model to ONNX with graph and numerical verification."""
    try:
        model_wrapper = default_registry.get(model)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Exporting model [bold]{model}[/bold] to {output}...[/cyan]")
    exporter = ONNXExporter(default_opset=opset)
    try:
        result = exporter.export(
            model_wrapper=model_wrapper,
            output_path=output,
            opset_version=opset,
            dynamic_batch=dynamic_batch,
            verify=verify,
        )
    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")
        raise typer.Exit(code=1)

    status = "✅ PASSED" if result.verified else "⚠️ VERIFICATION FAILED"
    table = Table(title="ONNX Export Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Output File", result.output_path)
    table.add_row("File Size", f"{result.file_size_mb:.2f} MB")
    table.add_row("ONNX Opset", str(result.opset_version))
    table.add_row("Node Count", str(result.node_count))
    table.add_row("Dynamic Batch", str(result.dynamic_batch))
    table.add_row("Max Abs Error vs PyTorch", f"{result.max_absolute_error_vs_torch:.2e}")
    table.add_row("Parity Verification", status)

    console.print(table)


@app.command("benchmark")
def benchmark(
    model: str | None = typer.Option(None, "--model", "-m", help="Registered PyTorch model ID"),
    onnx_path: Path | None = typer.Option(None, "--onnx", help="Path to exported ONNX model"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size for benchmark"),
    warmup: int = typer.Option(10, "--warmup", "-w", help="Number of warmup iterations"),
    runs: int = typer.Option(50, "--runs", "-r", help="Number of benchmark iterations"),
    device: str = typer.Option("cpu", "--device", "-d", help="Execution device (cpu, cuda, mps)"),
    json_out: Path | None = typer.Option(None, "--json-out", "-o", help="Optional path to save benchmark JSON report"),
) -> None:
    """Execute reproducible latency, throughput, and memory benchmark on PyTorch or ONNX."""
    if model is None and onnx_path is None:
        console.print("[red]Error:[/red] Specify either --model <id> or --onnx <path>")
        raise typer.Exit(code=1)

    runner = BenchmarkRunner()

    if onnx_path is not None:
        console.print(f"[cyan]Benchmarking ONNX model at {onnx_path}...[/cyan]")
        if not onnx_path.exists():
            console.print(f"[red]File not found:[/red] {onnx_path}")
            raise typer.Exit(code=1)
        onnx_wrapper = ONNXModelWrapper(onnx_path)
        report = runner.benchmark_onnx(
            onnx_wrapper=onnx_wrapper,
            batch_size=batch_size,
            num_warmup_runs=warmup,
            num_benchmark_runs=runs,
        )
    else:
        assert model is not None
        console.print(f"[cyan]Benchmarking PyTorch model [bold]{model}[/bold] on {device}...[/cyan]")
        try:
            model_wrapper = default_registry.get(model, device=device)
        except KeyError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

        report = runner.benchmark_pytorch(
            model_wrapper=model_wrapper,
            batch_size=batch_size,
            num_warmup_runs=warmup,
            num_benchmark_runs=runs,
            device=device,
        )

    console.print(report.to_rich_table())

    if json_out is not None:
        report.save_json(json_out)
        console.print(f"[green]Report saved to:[/green] {json_out}")


@app.command("compare")
def compare(
    baseline: Path = typer.Option(..., "--baseline", help="Path to baseline BenchmarkReport JSON"),
    optimized: Path = typer.Option(..., "--optimized", help="Path to optimized BenchmarkReport JSON"),
    json_out: Path | None = typer.Option(None, "--json-out", "-o", help="Optional path to save comparison JSON"),
) -> None:
    """Compare baseline and optimized benchmark reports with fairness validation."""
    if not baseline.exists():
        console.print(f"[red]Baseline report not found:[/red] {baseline}")
        raise typer.Exit(code=1)
    if not optimized.exists():
        console.print(f"[red]Optimized report not found:[/red] {optimized}")
        raise typer.Exit(code=1)

    base_report = BenchmarkReport.load_json(baseline)
    opt_report = BenchmarkReport.load_json(optimized)

    comparison = BenchmarkComparison.compare(base_report, opt_report)
    console.print(comparison.to_rich_table())

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            f.write(comparison.model_dump_json(indent=2))
        console.print(f"[green]Comparison saved to:[/green] {json_out}")


if __name__ == "__main__":
    app()
