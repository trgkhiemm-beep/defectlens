"""Latency benchmark of every edge variant on THIS machine (batch 1, end to end).

  python scripts/benchmark.py --edge artifacts/edge --devices CPU GPU --embedders fp32 int8

Measures preprocessing + embedder + scorer on a real sample image for each (runtime,
device, embedder precision, bank size), plus load/compile time and on-disk size.

Fairness rules, each learned from a wrong number on an i5-1135G7:
  * One fresh PROCESS per configuration, PyTorch and OpenVINO never in the same process:
    merely importing openvino slowed PyTorch eager ~5x (151 -> 715 ms, clashing threading
    runtimes), which had inflated the reported speed-up to x61.
  * A cool-down before each configuration and a cap on measuring time: this 15 W laptop
    throttles hard (INT8 + 1% bank: 42 ms cool, 320 ms right after a heavy run).
    Numbers are "burst" latency; sustained full load is slower.
  * Every configuration is measured --repeats times in round-robin order; the table reports
    the median p50 and the min-max spread of p50 across repeats, so one throttled run
    cannot decide a headline number.
  * The PyTorch baseline uses a random bank of the same size: nearest-neighbour cost
    depends on bank SIZE, not values.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.edge.preprocess import preprocess  # noqa: E402  (no torch, no openvino)
from src.lineage import git_commit  # noqa: E402


# ------------------------------------------------------------------ workers
def measure(fn, runs: int, max_seconds: float, min_runs: int = 5) -> dict:
    samples, t0 = [], time.perf_counter()
    while len(samples) < runs and (len(samples) < min_runs or time.perf_counter() - t0 < max_seconds):
        samples.append(fn())
    col = lambda k: np.array([s[k] for s in samples])  # noqa: E731
    total = col("total")
    return {"p50_ms": float(np.percentile(total, 50)), "p95_ms": float(np.percentile(total, 95)),
            "fps": float(1000 / np.percentile(total, 50)), "runs": len(samples),
            "embed_p50_ms": float(np.median(col("embed"))), "score_p50_ms": float(np.median(col("score")))}


def worker_openvino(job: dict) -> dict:
    from src.edge.runtime import EdgeInspector
    image = cv2.imread(job["image"])
    precision, bank = job["variant"].split("/")
    insp = EdgeInspector(job["edge"], precision, bank, job["device"], categories=[job["category"]])
    for _ in range(job["warmup"]):
        insp.predict(image, job["category"])
    res = measure(lambda: insp.predict(image, job["category"])["timing_ms"], job["runs"], job["max_seconds"])
    info = insp.manifest["variants"][job["variant"]]["categories"][job["category"]]
    return {**res, "compile_ms": insp.compile_ms,
            "disk_mb": insp.manifest["embedders"][precision]["mb"] + info.get("scorer_mb", 0)}


def worker_torch(job: dict) -> dict:
    import torch
    from src.models.patchcore import Embedder, Scorer
    m, image = job["manifest"], cv2.imread(job["image"])
    size = m["image_size"]
    t0 = time.perf_counter()
    emb_model = Embedder(m["backbone"], tuple(m["layers"]), job["pretrained"]).eval()
    scorer = Scorer(torch.randn(job["bank_size"], emb_model.embed_dim), m["sigma"], size).eval()
    load_ms = (time.perf_counter() - t0) * 1000

    @torch.inference_mode()
    def once() -> dict:
        t0 = time.perf_counter()
        x = torch.from_numpy(preprocess(image, size))
        t1 = time.perf_counter()
        e = emb_model(x)
        t2 = time.perf_counter()
        scorer(e)
        t3 = time.perf_counter()
        return {"embed": (t2 - t1) * 1000, "score": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}

    for _ in range(job["warmup"]):
        once()
    res = measure(once, job["runs"], job["max_seconds"])
    params_mb = sum(p.numel() for p in emb_model.parameters()) * 4 / 2**20
    return {**res, "compile_ms": load_ms, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "disk_mb": round(params_mb + job["bank_size"] * emb_model.embed_dim * 4 / 2**20, 2)}


def run_job(job: dict, cooldown: float) -> dict:
    time.sleep(cooldown)
    out = subprocess.run([sys.executable, "-W", "ignore", __file__, "--worker", json.dumps(job)],
                         cwd=ROOT, capture_output=True, text=True)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
    if out.returncode or line is None:
        raise RuntimeError(f"worker failed for {job.get('variant', job.get('bank_size'))}:\n{out.stderr[-2000:]}")
    return json.loads(line[len("RESULT "):])


def device_names(devices: list[str]) -> dict:
    code = ("import json, openvino as ov; c = ov.Core(); "
            f"print(json.dumps({{d: c.get_property(d, 'FULL_DEVICE_NAME') "
            f"for d in c.available_devices if d in {devices!r}}}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------- orchestrator
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default=Path("artifacts/edge"), type=Path)
    ap.add_argument("--devices", nargs="*", default=["CPU", "GPU"])
    ap.add_argument("--category", help="default: first category that has scorers for every variant")
    ap.add_argument("--embedders", nargs="*", help="subset of embedder precisions, e.g. fp32 int8")
    ap.add_argument("--banks", nargs="*", help="subset of banks, e.g. r0.1 r0.01")
    ap.add_argument("--repeats", type=int, default=3, help="round-robin repetitions of every configuration")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=10.0, help="stop measuring a configuration after this")
    ap.add_argument("--cooldown", type=float, default=20.0, help="idle seconds before each configuration")
    ap.add_argument("--no-torch", action="store_true", help="skip the PyTorch eager baseline")
    ap.add_argument("--no-pretrained", action="store_true", help="tests only (random torch weights)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:  # child process: exactly one configuration, one runtime
        job = json.loads(args.worker)
        res = worker_torch(job) if job["runtime"] == "torch" else worker_openvino(job)
        print("RESULT " + json.dumps(res), flush=True)
        return

    manifest = json.loads((args.edge / "manifest.json").read_text())
    variants = {k: v for k, v in manifest["variants"].items()
                if (not args.embedders or v["precision"] in args.embedders)
                and (not args.banks or k.split("/")[1] in args.banks)}
    cat = args.category or next(c for c in next(iter(variants.values()))["categories"]
                                if all(v["categories"][c].get("scorer") for v in variants.values()))
    image = args.edge / "samples" / f"{cat}_defect.png"
    hw = device_names(args.devices)
    base = {"edge": str(args.edge), "category": cat, "image": str(image), "runs": args.runs,
            "warmup": args.warmup, "max_seconds": args.max_seconds}
    print(f"category={cat} devices={hw}", flush=True)

    configs = []
    if not args.no_torch:
        for key in [k for k, v in variants.items() if v["precision"] == "fp32"]:
            n = variants[key]["categories"][cat]["bank_size"]
            configs.append(({"runtime": "PyTorch eager", "device": "CPU", "embedder": "fp32",
                             "bank": key.split("/")[1], "bank_size": n},
                            {**base, "runtime": "torch", "manifest": manifest, "bank_size": n,
                             "pretrained": not args.no_pretrained}))
    for device in hw:
        for key, v in variants.items():
            configs.append(({"runtime": "OpenVINO", "device": device, "embedder": v["precision"],
                             "bank": key.split("/")[1], "bank_size": v["categories"][cat]["bank_size"],
                             "image_auroc_mean": float(np.mean([c["image_auroc"] for c in v["categories"].values()]))},
                            {**base, "runtime": "openvino", "variant": key, "device": device}))
    results: list[list[dict]] = [[] for _ in configs]
    for rep in range(args.repeats):  # round-robin: thermal drift spreads over all configurations
        for i, (label, job) in enumerate(configs):
            results[i].append(run_job(job, args.cooldown))
            print(f"repeat {rep + 1}/{args.repeats}", {k: label[k] for k in ("runtime", "device", "embedder", "bank")},
                  f"p50={results[i][-1]['p50_ms']:.1f} ms", flush=True)
    rows = []
    for (label, _), reps in zip(configs, results):
        best = sorted(reps, key=lambda r: r["p50_ms"])[len(reps) // 2]  # the median repeat
        p50s = [r["p50_ms"] for r in reps]
        rows.append({**label, **best, "p50_ms": float(np.median(p50s)), "p50_min_ms": min(p50s),
                     "p50_max_ms": max(p50s), "fps": float(1000 / np.median(p50s)), "repeats": len(reps)})

    out = {"category": cat, "hardware": hw, "platform": platform.platform(), "git_commit": git_commit(),
           "torch": next((r["torch"] for r in rows if "torch" in r), None), "runs": args.runs,
           "max_seconds": args.max_seconds, "cooldown_s": args.cooldown, "repeats": args.repeats, "results": rows}
    (args.edge / "benchmark.json").write_text(json.dumps(out, indent=2))
    write_markdown(out, args.edge / "benchmark.md")


def write_markdown(out: dict, path: Path) -> None:
    base = next((r for r in out["results"] if r["runtime"] == "PyTorch eager" and r["bank"] == "r0.1"), None)
    lines = [f"Hardware: {', '.join(f'{d}: {n}' for d, n in out['hardware'].items())}  ",
             f"Batch 1, 256x256, `{out['category']}` sample, end to end (preprocess + embed + score). "
             f"One process per configuration, {out['cooldown_s']:.0f} s cool-down before each, warm-up, then up to "
             f"{out['runs']} runs or {out['max_seconds']:.0f} s; {out.get('repeats', 1)} round-robin repeats, "
             "p50 = median over repeats [min-max]. Burst latency: a 15 W laptop is slower under sustained load.", "",
             "| Runtime | Device | Embedder | Bank | p50 (ms) [spread] | p95 (ms) | FPS | Embed / Score p50 (ms) | "
             "Speed-up vs PyTorch r0.1 | Mean img AUROC | Load/compile (s) | Disk (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        speed = f"x{base['p50_ms'] / r['p50_ms']:.1f}" if base else "-"
        auroc = f"{r['image_auroc_mean']:.4f}" if "image_auroc_mean" in r else "(= OV FP32)"
        lines.append(f"| {r['runtime']} | {r['device']} | {r['embedder'].upper()} | {r['bank']} ({r['bank_size']}) | "
                     f"{r['p50_ms']:.1f} [{r.get('p50_min_ms', r['p50_ms']):.0f}-{r.get('p50_max_ms', r['p50_ms']):.0f}] | "
                     f"{r['p95_ms']:.1f} | {r['fps']:.1f} | "
                     f"{r['embed_p50_ms']:.1f} / {r['score_p50_ms']:.1f} | {speed} | {auroc} | "
                     f"{r['compile_ms'] / 1000:.1f} | {r['disk_mb']:.1f} |")
    lines.append(f"\n_code `{out['git_commit']}`, torch {out['torch']}, {out['platform']}_")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
