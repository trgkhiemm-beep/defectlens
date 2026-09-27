"""Anomaly-detection metrics in plain numpy (no sklearn dependency).

  image_auroc / pixel_auroc : ranking quality, threshold-free
  aupro                     : per-region overlap integrated up to FPR 0.3 (MVTec AD standard);
                              unlike pixel AUROC it is not dominated by large defects
  threshold selection       : done on the VALIDATION split only; the test split is scored
                              with that fixed threshold, and the best achievable ("oracle")
                              test F1 is reported alongside to expose the selection gap
"""
from __future__ import annotations

import cv2
import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n1 = int(labels.sum())
    n0 = labels.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based average rank per unique value
    ranks = avg_rank[inverse]
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aupro(amaps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3, steps: int = 300) -> float:
    """Normalised area under the PRO curve for FPR in [0, max_fpr].

    amaps, masks: (N, H, W). For each threshold t:
      PRO(t) = mean over ground-truth regions of |region & (amap >= t)| / |region|
      FPR(t) = fraction of normal pixels with amap >= t
    Thresholds are quantiles of the normal-pixel scores, so the FPR grid is uniform.
    """
    amaps = np.asarray(amaps, np.float32)
    masks = np.asarray(masks).astype(bool)
    normal = np.sort(amaps[~masks])
    regions = []
    for amap, mask in zip(amaps, masks):
        if mask.any():
            n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            regions += [np.sort(amap[lab == k]) for k in range(1, n)]
    if not regions or normal.size == 0:
        return float("nan")

    fprs = np.linspace(0.0, max_fpr, steps)
    thresholds = np.quantile(normal, 1.0 - fprs)  # FPR(t) ~= fprs by construction
    actual_fpr = 1.0 - np.searchsorted(normal, thresholds, side="left") / normal.size
    pro = np.zeros(steps)
    for r in regions:
        pro += 1.0 - np.searchsorted(r, thresholds, side="left") / r.size
    pro /= len(regions)
    order = np.argsort(actual_fpr)
    x, y = np.clip(actual_fpr[order], 0, max_fpr), pro[order]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / max_fpr)  # trapezoid rule


def binary_stats(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(scores) >= threshold
    labels = np.asarray(labels).astype(bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "f1": f1, "precision": precision, "recall": recall,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Threshold maximising F1 (candidates = midpoints between sorted unique scores)."""
    s = np.unique(np.asarray(scores, np.float64))
    candidates = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]]) if s.size > 1 else s
    best = max((binary_stats(scores, labels, t) for t in candidates), key=lambda d: (d["f1"], -d["fpr"]))
    return best


def evaluate(val: dict, test: dict) -> dict:
    """val/test: {"scores": (N,), "labels": (N,), "amaps": (N,H,W), "masks": (N,H,W)}."""
    chosen = best_f1_threshold(val["scores"], val["labels"])
    at_val_thr = binary_stats(test["scores"], test["labels"], chosen["threshold"])
    oracle = best_f1_threshold(test["scores"], test["labels"])
    return {
        "image_auroc": auroc(test["scores"], test["labels"]),
        "pixel_auroc": auroc(test["amaps"], test["masks"]),
        "aupro_30": aupro(test["amaps"], test["masks"]),
        "val_image_auroc": auroc(val["scores"], val["labels"]),
        "threshold": chosen["threshold"],
        "test_at_val_threshold": at_val_thr,
        "test_oracle": oracle,
        "f1_gap_vs_oracle": oracle["f1"] - at_val_thr["f1"],
    }
