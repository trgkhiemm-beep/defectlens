"""Image preprocessing shared by every runtime. numpy + OpenCV only: importing OpenVINO
and PyTorch in the same process slows PyTorch ~5x on Windows (conflicting threading
runtimes, measured 151 -> 715 ms), so the benchmark keeps them in separate processes
and this module must stay free of both."""
from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """Same transform as training (prepare_data INTER_AREA resize + ImageNet normalisation)."""
    if image_bgr.shape[:2] != (size, size):
        image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy()
