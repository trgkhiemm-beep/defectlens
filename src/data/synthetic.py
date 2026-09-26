"""Synthetic defect generation (DRAEM / CutPaste inspired).

Used ONLY to build a validation set for threshold selection, so the real
MVTec test set is never touched before final reporting (no test leakage).
"""
from __future__ import annotations

import cv2
import numpy as np


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Rough object mask: pixels far from the median border colour (works for
    dark or light backgrounds). Returns uint8 {0,1}."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - np.median(border, axis=0), axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.erode(fg, np.ones((9, 9), np.uint8))  # keep defects inside the object


def smooth_noise_mask(h: int, w: int, rng: np.random.Generator,
                      area: tuple[float, float] = (0.005, 0.04)) -> np.ndarray:
    """Perlin-like blob: upsampled low-res noise, thresholded to a target area."""
    scale = int(rng.choice([8, 16, 32]))
    noise = rng.random((max(h // scale, 2), max(w // scale, 2))).astype(np.float32)
    noise = cv2.GaussianBlur(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
    target = rng.uniform(*area)
    return (noise >= np.quantile(noise, 1.0 - target)).astype(np.uint8)


def _scratch_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    pts = [rng.integers(0, [w, h])]
    for _ in range(int(rng.integers(2, 5))):
        pts.append(np.clip(pts[-1] + rng.integers(-w // 6, w // 6, size=2), 0, [w - 1, h - 1]))
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, int(rng.integers(1, 4)))
    return mask


def _texture_from(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Foreign texture: a shuffled, colour-shifted copy of the image itself
    (no external dataset needed)."""
    h, w = img.shape[:2]
    tex = np.roll(img, (int(rng.integers(h)), int(rng.integers(w))), axis=(0, 1))
    tex = cv2.rotate(tex, int(rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180])))
    tex = cv2.resize(tex, (w, h))
    shift = rng.integers(-60, 60, size=3)
    return np.clip(tex.astype(np.int16) + shift, 0, 255).astype(np.uint8)


def synthesize_defect(img: np.ndarray, rng: np.random.Generator, use_foreground: bool = False,
                      max_tries: int = 10) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (defective_image uint8 HxWx3, mask uint8 {0,1}, defect_kind)."""
    h, w = img.shape[:2]
    fg = foreground_mask(img) if use_foreground else np.ones((h, w), np.uint8)
    kind = str(rng.choice(["texture_blend", "scratch", "cutpaste"]))

    for _ in range(max_tries):
        mask = _scratch_mask(h, w, rng) if kind == "scratch" else smooth_noise_mask(h, w, rng)
        mask &= fg
        if mask.sum() >= 20:
            break
    else:  # object too small / fg failed -> fall back to unrestricted mask
        mask = smooth_noise_mask(h, w, rng)

    if kind == "cutpaste":
        dy, dx = rng.integers(-h // 4, h // 4), rng.integers(-w // 4, w // 4)
        source = np.roll(img, (int(dy), int(dx)), axis=(0, 1))
        beta = 1.0
    elif kind == "scratch":
        source = np.full_like(img, int(rng.choice([20, 235])))
        beta = rng.uniform(0.6, 0.9)
    else:
        source = _texture_from(img, rng)
        beta = rng.uniform(0.5, 1.0)

    m = mask[..., None].astype(np.float32) * beta
    out = img.astype(np.float32) * (1 - m) + source.astype(np.float32) * m
    return out.astype(np.uint8), mask, kind
