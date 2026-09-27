"""Latency benchmark of every edge variant on THIS machine (batch 1, end to end).

  python scripts/benchmark.py --edge artifacts/edge --devices CPU GPU

Measures preprocessing + embedder + scorer on a real sample image, after warm-up,
for each (runtime, device, embedder precision, bank size); plus model compile time
(cold start) and on-disk size. The PyTorch eager FP32 baseline uses a random bank of
the same size: nearest-neighbour cost depends on bank SIZE, not its values.
Plug the laptop in and close heavy apps: CPU turbo/thermal state moves the numbers.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.edge.runtime import EdgeInspector, preprocess  # noqa: E402
from src.lineage import git_commit  # noqa: E402
from src.models.patchcore import Embedder, Scorer  # noqa: E402


def summarize(samples: list[dict]) -> dict:
    col = lambda k: np.array([s[k] for s in samples])  # noqa: E731
    total = col("total")
    return {"p50_ms": float(np.percentile(total, 50)), "p95_ms": float(np.percentile(total, 95)),
            "fps": float(1000 / np.percentile(total, 50)),
            "embed_p50_ms": float(np.median(col("embed"))), "score_p50_ms": float(np.median(col("score")))}


def bench_openvino(edge: Path, key: str, cat: str, device: str, image: np.ndarray, runs: int, warmup: int) -> dict:
    precision, bank = key.split("/")
    insp = EdgeInspector(edge, precision, bank, device, categories=[cat])
    for _ in range(warmup):
        insp.predict(image, cat)
    res = summarize([insp.predict(image, cat)["timing_ms"] for _ in range(runs)])
    info = insp.manifest["variants"][key]["categories"][cat]
    res.update(compile_ms=insp.compile_ms,
               disk_mb=insp.manifest["embedders"][precision]["mb"] + info.get("scorer_mb", 0))
    return res


@torch.inference_mode()
def bench_torch(manifest: dict, bank_size: int, image: np.ndarray, runs: int, warmup: int, pretrained: bool) -> dict:
    size = manifest["image_size"]
    t0 = time.perf_counter()
    emb_model = Embedder(manifest["backbone"], tuple(manifest["layers"]), pretrained).eval()
    scorer = Scorer(torch.randn(bank_size, emb_model.embed_dim), manifest["sigma"], size).eval()
    load_ms = (time.perf_counter() - t0) * 1000

    def once() -> dict:
        t0 = time.perf_counter()
        x = torch.from_numpy(preprocess(image, size))
        t1 = time.perf_counter()
        e = emb_model(x)
        t2 = time.perf_counter()
        scorer(e)
        t3 = time.perf_counter()
        return {"embed": (t2 - t1) * 1000, "score": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}

    for _ in range(warmup):
        once()
    res = summarize([once() for _ in range(runs)])
    params_mb = sum(p.numel() for p in emb_model.parameters()) * 4 / 2**20
    res.update(compile_ms=load_ms, disk_mb=round(params_mb + bank_size * emb_model.embed_dim * 4 / 2**20, 2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default=Path("artifacts/edge"), type=Path)
    ap.add_argument("--devices", nargs="*", default=["CPU", "GPU"])
    ap.add_argument("--category", help="default: first category that has scorers for every variant")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-torch", action="store_true", help="skip the PyTorch eager baseline")
    ap.add_argument("--no-pretrained", action="store_true", help="tests only (random torch weights)")
    args = ap.parse_args()

    manifest = json.loads((args.edge / "manifest.json").read_text())
    variants = manifest["variants"]
    cat = args.category or next(c for c in next(iter(variants.values()))["categories"]
                                if all(v["categories"][c].get("scorer") for v in variants.values()))
    sample = args.edge / "samples" / f"{cat}_defect.png"
    image = cv2.imread(str(sample)) if sample.exists() else \
        np.random.default_rng(0).integers(0, 255, (manifest["image_size"],) * 2 + (3,), dtype=np.uint8)
    core = ov.Core()
    devices = [d for d in args.devices if d in core.available_devices]
    hw = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices}
    print(f"category={cat} devices={hw} runs={args.runs}", flush=True)

    rows = []
    if not args.no_torch:
        for key in [k for k in variants if k.startswith("fp32/")]:
            n = variants[key]["categories"][cat]["bank_size"]
            r = bench_torch(manifest, n, image, args.runs, args.warmup, not args.no_pretrained)
            rows.append({"runtime": "PyTorch eager", "device": "CPU", "embedder": "fp32",
                         "bank": key.split("/")[1], "bank_size": n, **r})
            print(rows[-1], flush=True)
    for device in devices:
        for key, v in variants.items():
            r = bench_openvino(args.edge, key, cat, device, image, args.runs, args.warmup)
            rows.append({"runtime": "OpenVINO", "device": device, "embedder": v["precision"],
                         "bank": key.split("/")[1], "bank_size": v["categories"][cat]["bank_size"],
                         "image_auroc_mean": float(np.mean([c["image_auroc"] for c in v["categories"].values()])),
                         **r})
            print({k: rows[-1][k] for k in ("runtime", "device", "embedder", "bank", "p50_ms", "fps")}, flush=True)

    out = {"category": cat, "hardware": hw, "platform": platform.platform(), "git_commit": git_commit(),
           "openvino": ov.get_version(), "torch": torch.__version__, "runs": args.runs, "results": rows}
    (args.edge / "benchmark.json").write_text(json.dumps(out, indent=2))
    write_markdown(out, args.edge / "benchmark.md")


def write_markdown(out: dict, path: Path) -> None:
    base = next((r for r in out["results"] if r["runtime"] == "PyTorch eager" and r["bank"] == "r0.1"), None)
    lines = [f"Hardware: {', '.join(f'{d}: {n}' for d, n in out['hardware'].items())}  ",
             f"Batch 1, 256x256, `{out['category']}` sample, p50 over {out['runs']} runs after warm-up, "
             f"end to end (preprocess + embed + score).", "",
             "| Runtime | Device | Embedder | Bank | p50 (ms) | p95 (ms) | FPS | Embed / Score p50 (ms) | "
             "Speed-up | Mean img AUROC | Load/compile (s) | Disk (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        speed = f"x{base['p50_ms'] / r['p50_ms']:.1f}" if base else "-"
        auroc = f"{r['image_auroc_mean']:.4f}" if "image_auroc_mean" in r else "(= fp32 OV)"
        lines.append(f"| {r['runtime']} | {r['device']} | {r['embedder'].upper()} | {r['bank']} ({r['bank_size']}) | "
                     f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps']:.1f} | "
                     f"{r['embed_p50_ms']:.1f} / {r['score_p50_ms']:.1f} | {speed} | {auroc} | "
                     f"{r['compile_ms'] / 1000:.1f} | {r['disk_mb']:.1f} |")
    lines.append(f"\n_code `{out['git_commit']}`, OpenVINO {out['openvino']}, torch {out['torch']}, {out['platform']}_")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
