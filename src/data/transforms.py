"""Albumentations pipelines: category-aware train aug, eval, robustness suite.

Requires albumentations>=2.0 (parameter names changed from 1.x).
"""
from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # skip network version check (slow/offline)

import albumentations as A  # noqa: E402
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _finalize(size: int) -> list:
    return [A.Resize(size, size), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()]


def build_train_transform(size: int, aug: dict) -> A.Compose:
    """Light, physically plausible augmentation of NORMAL images only."""
    ops = []
    if aug.get("hflip"):
        ops.append(A.HorizontalFlip(p=0.5))
    if aug.get("vflip"):
        ops.append(A.VerticalFlip(p=0.5))
    if aug.get("rot90"):
        ops.append(A.RandomRotate90(p=0.5))
    if aug.get("max_rotate", 0) > 0:
        r = aug["max_rotate"]
        ops.append(A.Affine(rotate=(-r, r), border_mode=cv2.BORDER_REPLICATE, p=0.5))
    # Mild lighting drift, as seen between shifts on a real production line.
    ops.append(A.RandomBrightnessContrast(0.05, 0.05, p=0.3))
    return A.Compose(ops + _finalize(size))


def build_eval_transform(size: int) -> A.Compose:
    return A.Compose(_finalize(size))


# Edge-case suite: each corruption simulates a real failure mode on the line.
# Evaluated AFTER training to measure AUROC drop -> robustness table in README.
CORRUPTIONS: dict[str, A.BasicTransform] = {
    "brightness_low": A.RandomBrightnessContrast((-0.3, -0.3), (0, 0), p=1),   # dim lighting
    "brightness_high": A.RandomBrightnessContrast((0.3, 0.3), (0, 0), p=1),    # glare
    "gaussian_blur": A.GaussianBlur(blur_limit=(5, 5), p=1),                   # out of focus
    "motion_blur": A.MotionBlur(blur_limit=(9, 9), p=1),                       # conveyor motion
    "sensor_noise": A.GaussNoise(std_range=(0.06, 0.06), p=1),                 # cheap camera
    "jpeg": A.ImageCompression(quality_range=(25, 25), p=1),                   # streaming compression
    "misalign": A.Affine(rotate=(4, 4), translate_percent=(0.03, 0.03),
                         border_mode=cv2.BORDER_REPLICATE, p=1),               # part not centred
}


def build_corruption_transform(name: str, size: int) -> A.Compose:
    return A.Compose([CORRUPTIONS[name]] + _finalize(size))
