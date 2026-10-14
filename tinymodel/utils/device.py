"""Device utilities for TinyModel Lab."""

import torch


def get_available_device(prefer_cpu: bool = False) -> torch.device:
    """Return the optimal available hardware device.

    Priority:
    1. CPU (if prefer_cpu is True)
    2. CUDA (NVIDIA GPUs)
    3. MPS (Apple Silicon GPUs)
    4. CPU (Fallback)
    """
    if prefer_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device: torch.device | str) -> None:
    """Synchronize the specified device to guarantee all queued operations complete.

    Critical for accurate and reproducible latency benchmarking.
    """
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_obj.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
