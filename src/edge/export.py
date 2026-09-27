"""PyTorch -> OpenVINO IR conversion and NNCF post-training INT8 quantisation."""
from __future__ import annotations

from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch


def to_openvino(module: torch.nn.Module, example: torch.Tensor) -> ov.Model:
    """Trace a module with a fixed input shape (static shapes -> fastest CPU/iGPU kernels)."""
    with torch.no_grad():
        return ov.convert_model(module.eval(), example_input=example, input=[tuple(example.shape)])


def quantize_int8(model: ov.Model, calibration: list[np.ndarray], subset_size: int = 300) -> ov.Model:
    """Post-training INT8 (weights + activations) calibrated on NORMAL images only: the
    same data the model sees in production, and no labels are needed."""
    return nncf.quantize(model, nncf.Dataset(calibration), subset_size=min(subset_size, len(calibration)),
                         fast_bias_correction=True)


def save(model: ov.Model, path: Path, fp16_weights: bool) -> float:
    """Write .xml/.bin; returns size of the .bin in MB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(path), compress_to_fp16=fp16_weights)
    return round(path.with_suffix(".bin").stat().st_size / 2**20, 2)
