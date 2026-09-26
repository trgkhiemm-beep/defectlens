from __future__ import annotations

import cv2
import numpy as np

# Bump whenever generation logic changes; recorded in dataset_meta.json.
SYNTHETIC_VERSION = 2


def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Object mask for parts on a UNIFORM background (capsule, metal_nut):
    pixels far from the median border colour. Returns uint8 {0,1}.
    Not suitable for cluttered backgrounds (e.g. transistor on a PCB) -> use ROI."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - np.median(border, axis=0), axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, fg = cv2.threshold(dist, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.erode(fg, np.ones((9, 9), np.uint8))  # keep defects inside the object


def roi_mask(h: int, w: int, roi: list[float] | None) -> np.ndarray:
    """Inspection ROI in normalised [x0, y0, x1, y1]; None = whole image.
    Aligned fixtures + fixed ROI is exactly how production AOI stations work."""
    m = np.zeros((h, w), np.uint8)
    if roi is None:
        m[:] = 1
    else:
        x0, y0, x1, y1 = roi
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1
    return m


def inspection_region(img: np.ndarray, foreground: str = "none", roi: list[float] | None = None) -> np.ndarray:
    region = roi_mask(*img.shape[:2], roi)
    if foreground == "color":
        fg = foreground_mask(img) & region
        if fg.sum() > 0.02 * region.sum():  # guard against a failed Otsu split
            region = fg
    return region


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def blob_mask(region: np.ndarray, rng: np.random.Generator,
              area: tuple[float, float] = (0.01, 0.06)) -> np.ndarray:
    """One Perlin-like blob inside `region`; area is a fraction of the region."""
    h, w = region.shape
    scale = int(rng.choice([16, 32]))
    noise = rng.random((max(h // scale, 2), max(w // scale, 2))).astype(np.float32)
    noise = cv2.GaussianBlur(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
    inside = region.astype(bool)
    thr = np.quantile(noise[inside], 1.0 - rng.uniform(*area))
    return _largest_component(((noise >= thr) & inside).astype(np.uint8))


def scratch_mask(region: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A short polyline starting inside the region, clipped to it."""
    h, w = region.shape
    ys, xs = np.nonzero(region)
    k = int(rng.integers(len(ys)))
    pts = [np.array([xs[k], ys[k]])]
    step = max(min(h, w) // 10, 4)
    for _ in range(int(rng.integers(2, 4))):
        pts.append(np.clip(pts[-1] + rng.integers(-step, step + 1, size=2), 0, [w - 1, h - 1]))
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 1, int(rng.integers(2, 4)))
    return mask & region


def _texture_from(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Foreign texture: a shifted, rotated, colour-shifted copy of the image
    itself (no external texture dataset needed)."""
    h, w = img.shape[:2]
    tex = np.roll(img, (int(rng.integers(h)), int(rng.integers(w))), axis=(0, 1))
    tex = cv2.rotate(tex, int(rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180])))
    tex = cv2.resize(tex, (w, h))
    shift = rng.integers(-60, 60, size=3)
    return np.clip(tex.astype(np.int16) + shift, 0, 255).astype(np.uint8)


def synthesize_defect(img: np.ndarray, rng: np.random.Generator, region: np.ndarray | None = None,
                      min_pixels: int = 30, max_tries: int = 10) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (defective_image uint8 HxWx3, mask uint8 {0,1}, defect_kind)."""
    h, w = img.shape[:2]
    region = np.ones((h, w), np.uint8) if region is None else region
    kind = str(rng.choice(["texture_blend", "scratch", "cutpaste"]))

    for _ in range(max_tries):
        mask = scratch_mask(region, rng) if kind == "scratch" else blob_mask(region, rng)
        if mask.sum() >= min_pixels:
            break
    else:  # tiny region: fall back to a guaranteed-visible blob
        kind, mask = "texture_blend", blob_mask(region, rng, area=(0.08, 0.12))

    if kind == "cutpaste":
        dy, dx = rng.integers(-h // 4, h // 4), rng.integers(-w // 4, w // 4)
        source = np.roll(img, (int(dy), int(dx)), axis=(0, 1))
        beta = 1.0
    elif kind == "scratch":
        local = img[mask.astype(bool)].mean()
        source = np.full_like(img, 230 if local < 110 else 25)  # contrast with the surface
        beta = rng.uniform(0.6, 0.9)
    else:
        source = _texture_from(img, rng)
        beta = rng.uniform(0.5, 1.0)

    m = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)[..., None] * beta  # soft edges
    out = img.astype(np.float32) * (1 - m) + source.astype(np.float32) * m
    return out.astype(np.uint8), mask, kind
