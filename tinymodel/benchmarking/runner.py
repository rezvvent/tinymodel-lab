"""Benchmark runner executing rigorous, reproducible inference performance measurements."""

from __future__ import annotations

import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from tinymodel.benchmarking.hardware import HardwareInfo
from tinymodel.benchmarking.metrics import BenchmarkReport
from tinymodel.evaluation.metrics import evaluate_classifier
from tinymodel.exporters.onnx import ONNXModelWrapper
from tinymodel.models.base import ModelWrapper
from tinymodel.profiling.profiler import ModelProfiler
from tinymodel.utils.device import get_available_device, synchronize_device


class BenchmarkRunner:
    """Orchestrates reproducible latency, throughput, memory, and accuracy benchmarks."""

    def __init__(self, hardware: HardwareInfo | None = None) -> None:
        self.hardware = hardware or HardwareInfo.capture()

    def benchmark_pytorch(
        self,
        model_wrapper: ModelWrapper,
        batch_size: int = 1,
        num_warmup_runs: int = 10,
        num_benchmark_runs: int = 50,
        device: torch.device | str | None = None,
        eval_dataloader: DataLoader | None = None,
    ) -> BenchmarkReport:
        """Run benchmark on PyTorch model wrapper."""
        target_device = (
            torch.device(device) if device is not None else get_available_device(prefer_cpu=True)
        )
        model_wrapper.to(target_device)
        module = model_wrapper.module
        module.eval()

        # Measure load/init overhead
        t0_load = time.perf_counter_ns()
        dummy_input = model_wrapper.get_dummy_input(batch_size=batch_size, device=target_device)
        synchronize_device(target_device)
        load_time_ms = (time.perf_counter_ns() - t0_load) / 1e6

        # Profile structural properties
        profiler = ModelProfiler()
        profile = profiler.profile(model_wrapper, batch_size=batch_size)

        # Warm-up phase (discarded from statistics)
        with torch.inference_mode():
            for _ in range(num_warmup_runs):
                _ = module(dummy_input)
                synchronize_device(target_device)

        # Benchmark phase
        latencies_ms: list[float] = []

        is_cuda = target_device.type == "cuda" and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(target_device)

        process = psutil.Process()
        mem_before = process.memory_info().rss

        with torch.inference_mode():
            for _ in range(num_benchmark_runs):
                synchronize_device(target_device)
                t_start = time.perf_counter_ns()

                _ = module(dummy_input)

                synchronize_device(target_device)
                t_end = time.perf_counter_ns()
                latencies_ms.append((t_end - t_start) / 1e6)

        mem_after = process.memory_info().rss
        if is_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(target_device) / (1024 * 1024)
        else:
            peak_memory_mb = max(mem_after, mem_before) / (1024 * 1024)

        # Calculate latency statistics
        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        min_lat = float(np.min(lat_arr))
        max_lat = float(np.max(lat_arr))

        # Throughput = samples per second
        throughput = (batch_size * 1000.0) / mean_lat if mean_lat > 0 else 0.0

        # Optional quality evaluation
        accuracy, top5_acc, f1_mac, f1_wt = None, None, None, None
        if eval_dataloader is not None:
            metrics = evaluate_classifier(model_wrapper, eval_dataloader, device=target_device)
            accuracy = metrics.accuracy
            top5_acc = metrics.top5_accuracy
            f1_mac = metrics.f1_macro
            f1_wt = metrics.f1_weighted

        return BenchmarkReport(
            model_id=model_wrapper.metadata.model_id,
            model_type="pytorch",
            hardware=self.hardware,
            device=str(target_device),
            batch_size=batch_size,
            input_shape=list(model_wrapper.metadata.input_shape),
            num_warmup_runs=num_warmup_runs,
            num_benchmark_runs=num_benchmark_runs,
            total_parameters=profile.total_parameters,
            trainable_parameters=profile.trainable_parameters,
            logical_sparsity=profile.global_sparsity,
            model_size_mb=profile.total_size_mb,
            load_time_ms=round(load_time_ms, 3),
            latency_p50_ms=round(p50, 3),
            latency_p90_ms=round(p90, 3),
            latency_p95_ms=round(p95, 3),
            latency_p99_ms=round(p99, 3),
            latency_mean_ms=round(mean_lat, 3),
            latency_std_ms=round(std_lat, 3),
            latency_min_ms=round(min_lat, 3),
            latency_max_ms=round(max_lat, 3),
            throughput_samples_per_sec=round(throughput, 2),
            peak_memory_mb=round(peak_memory_mb, 2),
            accuracy=accuracy,
            top5_accuracy=top5_acc,
            f1_macro=f1_mac,
            f1_weighted=f1_wt,
            raw_latencies_ms=[round(x, 4) for x in latencies_ms],
        )

    def benchmark_onnx(
        self,
        onnx_wrapper: ONNXModelWrapper,
        input_shape: tuple[int, ...] = (3, 224, 224),
        model_id: str | None = None,
        batch_size: int = 1,
        num_warmup_runs: int = 10,
        num_benchmark_runs: int = 50,
        eval_dataloader: DataLoader | None = None,
    ) -> BenchmarkReport:
        """Run benchmark on ONNX Runtime model."""
        target_model_id = model_id or onnx_wrapper.model_path.stem

        # Generate dummy input matching input_shape
        dummy_input = np.random.randn(batch_size, *input_shape).astype(np.float32)

        # Warm-up phase
        for _ in range(num_warmup_runs):
            _ = onnx_wrapper.forward(dummy_input)

        # Benchmark phase
        latencies_ms: list[float] = []
        process = psutil.Process()
        mem_before = process.memory_info().rss

        for _ in range(num_benchmark_runs):
            t_start = time.perf_counter_ns()
            _ = onnx_wrapper.forward(dummy_input)
            t_end = time.perf_counter_ns()
            latencies_ms.append((t_end - t_start) / 1e6)

        mem_after = process.memory_info().rss
        peak_memory_mb = max(mem_after, mem_before) / (1024 * 1024)

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        min_lat = float(np.min(lat_arr))
        max_lat = float(np.max(lat_arr))

        throughput = (batch_size * 1000.0) / mean_lat if mean_lat > 0 else 0.0

        # Optional quality evaluation
        accuracy, top5_acc, f1_mac, f1_wt = None, None, None, None
        if eval_dataloader is not None:
            metrics = evaluate_classifier(onnx_wrapper, eval_dataloader, device="cpu")
            accuracy = metrics.accuracy
            top5_acc = metrics.top5_accuracy
            f1_mac = metrics.f1_macro
            f1_wt = metrics.f1_weighted

        return BenchmarkReport(
            model_id=target_model_id,
            model_type="onnx",
            hardware=self.hardware,
            device="onnx_runtime_" + "_".join(onnx_wrapper.active_providers),
            batch_size=batch_size,
            input_shape=list(input_shape),
            num_warmup_runs=num_warmup_runs,
            num_benchmark_runs=num_benchmark_runs,
            total_parameters=0,  # Parameters encapsulated inside ONNX graph
            trainable_parameters=0,
            logical_sparsity=0.0,
            model_size_mb=onnx_wrapper.file_size_mb,
            load_time_ms=0.0,
            latency_p50_ms=round(p50, 3),
            latency_p90_ms=round(p90, 3),
            latency_p95_ms=round(p95, 3),
            latency_p99_ms=round(p99, 3),
            latency_mean_ms=round(mean_lat, 3),
            latency_std_ms=round(std_lat, 3),
            latency_min_ms=round(min_lat, 3),
            latency_max_ms=round(max_lat, 3),
            throughput_samples_per_sec=round(throughput, 2),
            peak_memory_mb=round(peak_memory_mb, 2),
            accuracy=accuracy,
            top5_accuracy=top5_acc,
            f1_macro=f1_mac,
            f1_weighted=f1_wt,
            raw_latencies_ms=[round(x, 4) for x in latencies_ms],
        )
