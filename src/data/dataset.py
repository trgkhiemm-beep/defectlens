"""Backend-agnostic dataset driven by the manifest written by prepare_data.py.

Why not only anomalib's datamodule? The same loader must feed PyTorch, ONNX
Runtime and OpenVINO INT8 models (fair benchmark), plus NNCF calibration.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read_manifest(path: str | Path, category: str, split: str) -> list[dict]:
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["category"] == category and r["split"] == split]


class AnomalyDataset(Dataset):
    def __init__(self, root: str | Path, category: str, split: str, transform):
        self.root = Path(root)
        self.rows = read_manifest(self.root / "manifest.csv", category, split)
        if not self.rows:
            raise ValueError(f"No rows for category={category!r} split={split!r}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / r["image_path"])), cv2.COLOR_BGR2RGB)
        if r["mask_path"]:
            mask = (cv2.imread(str(self.root / r["mask_path"]), cv2.IMREAD_GRAYSCALE) > 127)
        else:
            mask = np.zeros(img.shape[:2], bool)
        out = self.transform(image=img, mask=mask.astype(np.uint8))
        return {
            "image": out["image"],                                # float32 3xHxW, normalised
            "mask": out["mask"].unsqueeze(0).float(),             # float32 1xHxW in {0,1}
            "label": torch.tensor(int(r["label"])),               # 0 normal, 1 anomalous
            "defect_type": r["defect_type"],
            "path": r["image_path"],
        }


def make_loader(ds: Dataset, batch_size: int = 32, shuffle: bool = False,
                num_workers: int = 2) -> DataLoader:
    # Images are pre-resized to 256px on disk, so decoding is cheap and 2-4
    # workers saturate a Kaggle T4. On Windows, call this under
    # `if __name__ == "__main__":` (spawn-based multiprocessing).
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)
