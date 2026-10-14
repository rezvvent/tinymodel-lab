"""Hardware and environment metadata capture for reproducible benchmarking."""

from __future__ import annotations

import platform

import psutil
import torch
from pydantic import BaseModel, Field


class HardwareInfo(BaseModel):
    """Snapshot of hardware, system, and library versions during execution."""

    os_name: str = Field(default_factory=lambda: platform.system())
    os_release: str = Field(default_factory=lambda: platform.release())
    machine: str = Field(default_factory=lambda: platform.machine())
    processor: str = Field(default_factory=lambda: platform.processor())
    python_version: str = Field(default_factory=lambda: platform.python_version())

    cpu_count_physical: int = Field(default_factory=lambda: psutil.cpu_count(logical=False) or 1)
    cpu_count_logical: int = Field(default_factory=lambda: psutil.cpu_count(logical=True) or 1)
    torch_num_threads: int = Field(default_factory=lambda: torch.get_num_threads())
    ram_total_bytes: int = Field(default_factory=lambda: psutil.virtual_memory().total)

    torch_version: str = Field(default_factory=lambda: torch.__version__)
    cuda_available: bool = Field(default_factory=lambda: torch.cuda.is_available())
    cuda_device_name: str | None = None
    cuda_vram_total_bytes: int | None = None
    mps_available: bool = Field(
        default_factory=lambda: hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    onnx_version: str | None = None
    onnxruntime_version: str | None = None

    @classmethod
    def capture(cls) -> HardwareInfo:
        """Capture current system and library state."""
        info = cls()

        # Capture CUDA if available
        if info.cuda_available:
            info.cuda_device_name = torch.cuda.get_device_name(0)
            info.cuda_vram_total_bytes = torch.cuda.get_device_properties(0).total_memory

        # Capture ONNX versions if installed
        try:
            import onnx

            info.onnx_version = onnx.__version__
        except ImportError:
            pass

        try:
            import onnxruntime

            info.onnxruntime_version = onnxruntime.__version__
        except ImportError:
            pass

        return info

    def is_comparable_with(self, other: HardwareInfo) -> tuple[bool, list[str]]:
        """Validate if another run was executed under comparable hardware conditions.

        Returns (is_comparable, list_of_discrepancies).
        """
        discrepancies = []
        if self.machine != other.machine:
            discrepancies.append(f"Architecture mismatch: {self.machine} vs {other.machine}")
        if self.cuda_available != other.cuda_available:
            discrepancies.append(
                f"CUDA mismatch: {self.cuda_available} vs {other.cuda_available}"
            )
        if self.mps_available != other.mps_available:
            discrepancies.append(f"MPS mismatch: {self.mps_available} vs {other.mps_available}")
        if self.torch_num_threads != other.torch_num_threads:
            discrepancies.append(
                f"CPU threads mismatch: {self.torch_num_threads} vs {other.torch_num_threads}"
            )

        return (len(discrepancies) == 0, discrepancies)
