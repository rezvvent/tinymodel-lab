"""Synthetic and standard dataset loaders for reproducible benchmarking and calibration."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    """Deterministic synthetic dataset for reproducible offline evaluation and calibration."""

    def __init__(
        self,
        num_samples: int = 100,
        input_shape: tuple[int, ...] = (3, 224, 224),
        num_classes: int = 10,
        is_text: bool = False,
        vocab_size: int = 2000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_text = is_text

        # Generate deterministic data
        g = torch.Generator().manual_seed(seed)
        if is_text:
            self.data = torch.randint(
                1, vocab_size, (num_samples, *input_shape), generator=g, dtype=torch.long
            )
        else:
            self.data = torch.randn(
                (num_samples, *input_shape), generator=g, dtype=torch.float32
            )

        self.targets = torch.randint(
            0, num_classes, (num_samples,), generator=g, dtype=torch.long
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def create_synthetic_dataloader(
    num_samples: int = 100,
    batch_size: int = 16,
    input_shape: tuple[int, ...] = (3, 224, 224),
    num_classes: int = 10,
    is_text: bool = False,
    vocab_size: int = 2000,
    seed: int = 42,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader serving deterministic synthetic data."""
    dataset = SyntheticDataset(
        num_samples=num_samples,
        input_shape=input_shape,
        num_classes=num_classes,
        is_text=is_text,
        vocab_size=vocab_size,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
