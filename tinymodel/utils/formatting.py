"""Formatting helpers for sizes, numbers, and latencies."""


def format_bytes(size_bytes: int | float) -> str:
    """Format bytes to human readable format (B, KB, MB, GB)."""
    if size_bytes < 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_params(num_params: int) -> str:
    """Format parameter count with standard metric prefixes (K, M, B)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def format_latency(latency_ms: float) -> str:
    """Format latency into human readable string with appropriate unit."""
    if latency_ms < 1.0:
        return f"{latency_ms * 1000.0:.2f} µs"
    if latency_ms >= 1000.0:
        return f"{latency_ms / 1000.0:.2f} s"
    return f"{latency_ms:.2f} ms"
