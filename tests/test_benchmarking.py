"""Unit tests for BenchmarkRunner, BenchmarkComparison, and evaluation metrics."""

from tinymodel.benchmarking.hardware import HardwareInfo
from tinymodel.benchmarking.metrics import BenchmarkComparison
from tinymodel.benchmarking.runner import BenchmarkRunner
from tinymodel.evaluation.metrics import evaluate_classifier
from tinymodel.exporters.onnx import ONNXExporter, ONNXModelWrapper


def test_hardware_info_capture():
    """Verify system and hardware information capture."""
    hw = HardwareInfo.capture()
    assert hw.os_name != ""
    assert hw.cpu_count_logical >= 1
    assert hw.torch_num_threads >= 1
    assert hw.ram_total_bytes > 0
    assert hw.torch_version != ""

    # Self-comparison must be 100% comparable
    is_comp, discrepancies = hw.is_comparable_with(hw)
    assert is_comp is True
    assert len(discrepancies) == 0


def test_benchmark_pytorch_runner(tiny_convnet, synthetic_dataloader):
    """Verify PyTorch model benchmarking produces valid latency and throughput."""
    runner = BenchmarkRunner()
    report = runner.benchmark_pytorch(
        model_wrapper=tiny_convnet,
        batch_size=4,
        num_warmup_runs=3,
        num_benchmark_runs=10,
        device="cpu",
        eval_dataloader=synthetic_dataloader,
    )

    assert report.model_id == "tiny_convnet"
    assert report.model_type == "pytorch"
    assert report.batch_size == 4
    assert len(report.raw_latencies_ms) == 10
    # Latencies must be sorted logically across percentiles
    assert report.latency_min_ms <= report.latency_p50_ms <= report.latency_max_ms
    assert report.latency_p50_ms <= report.latency_p95_ms <= report.latency_p99_ms
    assert report.throughput_samples_per_sec > 0
    assert report.peak_memory_mb > 0
    assert report.accuracy is not None
    assert 0.0 <= report.accuracy <= 1.0


def test_benchmark_onnx_runner(tmp_path, tiny_convnet, synthetic_dataloader):
    """Verify ONNX model benchmarking produces valid metrics."""
    onnx_file = tmp_path / "tiny.onnx"
    ONNXExporter(default_opset=18).export(tiny_convnet, onnx_file)

    runner = BenchmarkRunner()
    onnx_wrapper = ONNXModelWrapper(onnx_file)
    report = runner.benchmark_onnx(
        onnx_wrapper=onnx_wrapper,
        input_shape=(3, 32, 32),
        batch_size=4,
        num_warmup_runs=3,
        num_benchmark_runs=10,
        eval_dataloader=synthetic_dataloader,
    )

    assert report.model_type == "onnx"
    assert report.latency_p50_ms > 0
    assert report.throughput_samples_per_sec > 0
    assert report.accuracy is not None


def test_fair_benchmark_comparison(tmp_path, tiny_convnet):
    """Verify comparison between PyTorch and ONNX models and fairness check."""
    runner = BenchmarkRunner()
    pt_report = runner.benchmark_pytorch(
        model_wrapper=tiny_convnet,
        batch_size=1,
        num_warmup_runs=2,
        num_benchmark_runs=5,
        device="cpu",
    )

    onnx_file = tmp_path / "tiny.onnx"
    ONNXExporter(default_opset=18).export(tiny_convnet, onnx_file)
    onnx_wrapper = ONNXModelWrapper(onnx_file)
    onnx_report = runner.benchmark_onnx(
        onnx_wrapper=onnx_wrapper,
        input_shape=(3, 32, 32),
        batch_size=1,
        num_warmup_runs=2,
        num_benchmark_runs=5,
    )

    comparison = BenchmarkComparison.compare(pt_report, onnx_report)
    assert comparison.compression_ratio > 0
    assert comparison.speedup_ratio > 0
    assert comparison.throughput_gain_ratio > 0


def test_unfair_benchmark_comparison_detection(tiny_convnet):
    """Verify detection and warning when comparing benchmarks with mismatched conditions."""
    runner = BenchmarkRunner()
    rep1 = runner.benchmark_pytorch(tiny_convnet, batch_size=1, num_warmup_runs=1, num_benchmark_runs=2)
    rep2 = runner.benchmark_pytorch(tiny_convnet, batch_size=4, num_warmup_runs=1, num_benchmark_runs=2)

    comp = BenchmarkComparison.compare(rep1, rep2)
    assert comp.fair_comparison is False
    assert any("Batch size mismatch" in w for w in comp.discrepancy_warnings)


def test_evaluate_classifier_metrics(tiny_convnet, synthetic_dataloader):
    """Verify evaluation calculates real accuracy and F1 scores."""
    metrics = evaluate_classifier(tiny_convnet, synthetic_dataloader, device="cpu")
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert 0.0 <= metrics.f1_weighted <= 1.0
    assert metrics.num_samples == 20
