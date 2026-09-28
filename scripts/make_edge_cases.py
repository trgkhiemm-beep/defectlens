"""Generate edge-case inputs from a known-good sample, to test input validation (API) and
robustness (model). See docs/TESTING.md for the expected behaviour of each file.

  python scripts/make_edge_cases.py --sample deploy/model/samples/transistor_good.png --out test_inputs/edge_cases
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=Path("deploy/model/samples/transistor_good.png"), type=Path)
    ap.add_argument("--out", default=Path("test_inputs/edge_cases"), type=Path)
    ap.add_argument("--no-large", action="store_true", help="skip the ~100 MB over-limit file")
    args = ap.parse_args()
    img = cv2.imread(str(args.sample), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.sample}")
    o = args.out
    o.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # input-validation cases (API must answer 400, never 500)
    (o / "empty.png").write_bytes(b"")
    (o / "not_an_image.txt").write_text("hello, I am not an image")
    (o / "truncated.png").write_bytes(cv2.imencode(".png", img)[1].tobytes()[:200])
    if not args.no_large:  # > 10 MB upload limit
        cv2.imwrite(str(o / "too_large_over_10mb.bmp"), cv2.resize(img, (6000, 6000), interpolation=cv2.INTER_NEAREST))

    # format cases (must be accepted, HTTP 200). JPEG / upscaled / alpha / non-square keep the sample's
    # verdict; grayscale and 32 px do NOT (measured: DEFECT 1.13x and 1.86x), a documented limitation.
    cv2.imwrite(str(o / "grayscale.png"), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(str(o / "with_alpha.png"), cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
    cv2.imwrite(str(o / "as_jpeg_q90.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(o / "upscaled_1024.png"), cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(str(o / "tiny_32px.png"), cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(o / "non_square_wide.png"), cv2.resize(img, (512, 256)))

    # robustness cases (conditions the model never saw; verdicts may flip: that is the point)
    cv2.imwrite(str(o / "dark_lighting.png"), cv2.convertScaleAbs(img, alpha=0.6, beta=0))
    cv2.imwrite(str(o / "glare.png"), cv2.convertScaleAbs(img, alpha=1.0, beta=60))
    cv2.imwrite(str(o / "blurred.png"), cv2.GaussianBlur(img, (0, 0), 3))
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(o / "sensor_noise.png"), noisy)
    h, w = img.shape[:2]
    cv2.imwrite(str(o / "rotated_10deg.png"), cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1), (w, h),
                                                             borderMode=cv2.BORDER_REFLECT))
    cv2.imwrite(str(o / "shifted_20px.png"), np.roll(img, 20, axis=1))

    # out-of-distribution (not the selected part at all -> expect a warning)
    cv2.imwrite(str(o / "random_noise.png"), rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    cv2.imwrite(str(o / "flat_grey.png"), np.full((256, 256, 3), 128, np.uint8))
    for p in sorted(o.iterdir()):
        print(f"{p.stat().st_size / 1024:10.1f} KB  {p.name}")


if __name__ == "__main__":
    main()
