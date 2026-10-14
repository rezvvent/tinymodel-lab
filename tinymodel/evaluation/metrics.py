"""Model quality evaluation and classification metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from tinymodel.exporters.onnx import ONNXModelWrapper
from tinymodel.models.base import ModelWrapper


class ClassificationMetrics(BaseModel):
    """Classification quality metrics."""

    accuracy: float
    top5_accuracy: float | None = None
    f1_macro: float
    f1_weighted: float
    loss: float | None = None
    num_samples: int


def evaluate_classifier(
    model: ModelWrapper | nn.Module | ONNXModelWrapper,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    criterion: nn.Module | None = None,
) -> ClassificationMetrics:
    """Evaluate classification model on a DataLoader, computing real Accuracy, Top-5, and F1."""
    device_obj = torch.device(device) if isinstance(device, str) else device
    all_preds: list[int] = []
    all_targets: list[int] = []
    top5_correct = 0
    total_loss = 0.0
    total_samples = 0

    is_onnx = isinstance(model, ONNXModelWrapper)
    is_wrapper = isinstance(model, ModelWrapper)
    raw_module = model.module if is_wrapper else model

    if not is_onnx:
        assert isinstance(raw_module, nn.Module)
        raw_module.to(device_obj)
        raw_module.eval()

    if criterion is None and not is_onnx:
        criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for inputs, targets in dataloader:
            batch_sz = targets.size(0)
            total_samples += batch_sz
            targets_cpu = targets.cpu().numpy()
            all_targets.extend(targets_cpu.tolist())

            if is_onnx:
                # Inference with ONNX Runtime
                ort_outs = model.forward(inputs)
                logits_np = ort_outs[0]
                preds = np.argmax(logits_np, axis=1)
                all_preds.extend(preds.tolist())

                if logits_np.shape[1] >= 5:
                    top5_idx = np.argsort(logits_np, axis=1)[:, -5:]
                    for t, pred_top5 in zip(targets_cpu, top5_idx, strict=False):
                        if t in pred_top5:
                            top5_correct += 1
            else:
                inputs_dev = inputs.to(device_obj)
                targets_dev = targets.to(device_obj)
                logits = raw_module(inputs_dev)

                if criterion is not None:
                    loss_val = criterion(logits, targets_dev).item()
                    total_loss += loss_val * batch_sz

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())

                if logits.size(1) >= 5:
                    top5 = torch.topk(logits, k=5, dim=1).indices.cpu().numpy()
                    for t, pred_top5 in zip(targets_cpu, top5, strict=False):
                        if t in pred_top5:
                            top5_correct += 1

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    f1_mac = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    f1_wt = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    top5_acc = float(top5_correct / total_samples) if dataloader.dataset[0][1] is not None and len(all_preds) > 0 and len(np.unique(all_targets)) >= 5 else None
    avg_loss = float(total_loss / total_samples) if total_samples > 0 and not is_onnx else None

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        top5_accuracy=round(top5_acc, 4) if top5_acc is not None else None,
        f1_macro=round(f1_mac, 4),
        f1_weighted=round(f1_wt, 4),
        loss=round(avg_loss, 4) if avg_loss is not None else None,
        num_samples=total_samples,
    )
