"""Fit + evaluate PatchCore for every category of a validated dataset.

  python scripts/train_patchcore.py --data /kaggle/working/data/processed --out /kaggle/working/artifacts/patchcore

Per category -> <out>/<category>/
  model.pt       memory bank + model config (backbone weights are re-downloaded, frozen)
  metrics.json   AUROC / AUPRO, operating points of every threshold policy (chosen on
                 VAL only) vs the test oracle, val/test normal-score shift,
                 per-defect-type AUROC, latency, full lineage (dataset run, git commit)
  scores.json    image-level val/test scores -> re-analyse thresholds on CPU without
                 re-training (scripts/analyze_thresholds.py)
  examples.png   heatmaps of test defects, coloured relative to the chosen threshold
Across categories -> <out>/summary.json, <out>/summary.md (README-ready table)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import AnomalyDataset, make_loader  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.data.validate import load_ready_dataset  # noqa: E402
from src.eval.metrics import auroc, evaluate  # noqa: E402
from src.lineage import git_commit  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402


@torch.inference_mode()
def predict(model: PatchCore, loader, device) -> dict:
    out = {k: [] for k in ("scores", "labels", "amaps", "masks", "defect_types", "paths")}
    for b in loader:
        s, a = model(b["image"].to(device))
        out["scores"].append(s.cpu().numpy())
        out["amaps"].append(a[:, 0].cpu().numpy())
        out["labels"].append(b["label"].numpy())
        out["masks"].append(b["mask"][:, 0].numpy().astype(bool))
        out["defect_types"] += list(b["defect_type"])
        out["paths"] += list(b["path"])
    return {k: (np.concatenate(v) if k in ("scores", "labels", "amaps", "masks") else v) for k, v in out.items()}


@torch.inference_mode()
def latency_ms(model: PatchCore, size: int, device, runs: int) -> dict:
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(3):
        model(x)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"device": str(device), "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def per_defect_auroc(test: dict) -> dict:
    """Image AUROC of each defect type vs all normal test images -> shows WHICH defects fail."""
    normal = test["labels"] == 0
    types = np.array(test["defect_types"])
    return {t: auroc(np.r_[test["scores"][normal], test["scores"][types == t]],
                     np.r_[np.zeros(normal.sum()), np.ones((types == t).sum())])
            for t in sorted(set(types[~normal]))}


def save_examples(test: dict, data: Path, threshold: float, path: Path, n: int = 6) -> None:
    """One example per defect type: image | heatmap overlay (red = above threshold)."""
    picked, seen = [], set()
    for i, t in enumerate(test["defect_types"]):
        if test["labels"][i] == 1 and t not in seen:
            picked.append(i)
            seen.add(t)
    tiles = []
    for i in picked[:n]:
        img = cv2.imread(str(data / test["paths"][i]))
        heat = np.clip(test["amaps"][i] / (2 * threshold), 0, 1)  # 0.5 on the colour scale == threshold
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        over = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
        cnts, _ = cv2.findContours(test["masks"][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, (255, 255, 255), 1)  # white = ground truth
        verdict = "DEFECT" if test["scores"][i] >= threshold else "missed"
        cv2.putText(over, f"{test['defect_types'][i]}: {verdict}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        tiles.append(np.hstack([img, over]))
    if tiles:
        cv2.imwrite(str(path), np.vstack(tiles))


def run_category(cat: str, args, cfg: dict, meta: dict, device) -> dict:
    torch.manual_seed(cfg["seed"])
    size = meta["config"]["image_size"]
    tf = build_eval_transform(size)  # PatchCore stores raw normal features: no augmentation
    loader = lambda split: make_loader(AnomalyDataset(args.data, cat, split, tf),  # noqa: E731
                                       batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    model = PatchCore(cfg["backbone"], tuple(cfg["layers"]), not args.no_pretrained,
                      cfg["coreset_ratio"], cfg["sigma"])
    t0 = time.perf_counter()
    fit_stats = model.fit(loader("train"), device, cfg.get("max_bank"), cfg["seed"])
    fit_stats["fit_seconds"] = round(time.perf_counter() - t0, 1)

    val, test = predict(model, loader("val"), device), predict(model, loader("test"), device)
    metrics = evaluate(val, test, cfg["threshold_policy"], cfg["target_fpr"])
    metrics["per_defect_auroc"] = per_defect_auroc(test)
    metrics["latency_bs1"] = latency_ms(model, size, device, cfg["latency_runs"])

    out = args.out / cat
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state(), out / "model.pt")
    (out / "scores.json").write_text(json.dumps({
        split: {"scores": d["scores"].tolist(), "labels": d["labels"].tolist(),
                "defect_types": d["defect_types"], "paths": d["paths"]}
        for split, d in (("val", val), ("test", test))}))
    save_examples(test, args.data, metrics["threshold"], out / "examples.png")
    record = {"category": cat, "metrics": metrics, "fit": fit_stats, "model_config": model.config,
              "lineage": {"git_commit": git_commit(), "dataset_run_id": meta["run_id"],
                          "dataset_git_commit": meta["git_commit"], "device": str(device)}}
    (out / "metrics.json").write_text(json.dumps(record, indent=2))
    t, ops = metrics["test_at_threshold"], metrics["operating_points"]
    print(f"[{cat}] img_AUROC={metrics['image_auroc']:.4f} px_AUROC={metrics['pixel_auroc']:.4f} "
          f"AUPRO={metrics['aupro_30']:.4f} [{metrics['policy']}] F1={t['f1']:.3f} FPR={t['fpr']:.3f} "
          f"recall={t['recall']:.3f} (oracle F1 {ops['oracle']['f1']:.3f}) "
          f"bank={fit_stats['bank_size']} fit={fit_stats['fit_seconds']}s "
          f"lat_p50={metrics['latency_bs1']['p50_ms']:.1f}ms", flush=True)
    return record


def write_summary(records: list[dict], out: Path) -> None:
    (out / "summary.json").write_text(json.dumps(records, indent=2))
    policy = records[0]["metrics"]["policy"]
    target = records[0]["metrics"]["target_fpr"]
    rows = ["| Category | Image AUROC | Pixel AUROC | AUPRO@0.3 | Bank (MB) | p50 latency (ms) |",
            "|---|---|---|---|---|---|"]
    for r in records:
        m = r["metrics"]
        rows.append(f"| {r['category']} | {m['image_auroc']:.3f} | {m['pixel_auroc']:.3f} | {m['aupro_30']:.3f} | "
                    f"{r['fit']['bank_mb']} | {m['latency_bs1']['p50_ms']:.1f} |")
    mean = lambda k: np.mean([r["metrics"][k] for r in records])  # noqa: E731
    rows.append(f"| **mean** | **{mean('image_auroc'):.3f}** | **{mean('pixel_auroc'):.3f}** | "
                f"**{mean('aupro_30'):.3f}** | | |")

    rows += ["", f"Operating point (threshold chosen on val only; primary policy `{policy}`, target FPR {target}):", "",
             "| Category | val AUROC (synthetic) | F1 `val_f1` | FPR `val_f1` | F1 `normal_quantile` | "
             "FPR `normal_quantile` | Oracle F1 | test/val normal-score ratio |",
             "|---|---|---|---|---|---|---|---|"]
    for r in records:
        m, ops = r["metrics"], r["metrics"]["operating_points"]
        a, b = ops["policies"]["val_f1"], ops["policies"]["normal_quantile"]
        rows.append(f"| {r['category']} | {m['val_image_auroc']:.3f} | {a['f1']:.3f} | {a['fpr']:.3f} | "
                    f"{b['f1']:.3f} | {b['fpr']:.3f} | {ops['oracle']['f1']:.3f} | "
                    f"{ops['score_shift']['median_ratio_test_over_val']:.3f} |")
    lin = records[0]["lineage"]
    rows.append(f"\n_code `{lin['git_commit']}`, dataset run `{lin['dataset_run_id']}`, device `{lin['device']}`_")
    (out / "summary.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--out", default=Path("artifacts/patchcore"), type=Path)
    ap.add_argument("--config", default=ROOT / "configs/patchcore.yaml", type=Path)
    ap.add_argument("--categories", nargs="*", help="default: all categories in the dataset")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backbone", help="override config (e.g. resnet18 for a quick CPU run)")
    ap.add_argument("--coreset-ratio", type=float, help="override config")
    ap.add_argument("--no-pretrained", action="store_true", help="random backbone weights (tests only)")
    args = ap.parse_args()

    meta = load_ready_dataset(args.data)  # refuses unvalidated / stale datasets
    cfg = yaml.safe_load(args.config.read_text())
    if args.backbone:
        cfg["backbone"] = args.backbone
    if args.coreset_ratio:
        cfg["coreset_ratio"] = args.coreset_ratio
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"dataset run={meta['run_id']} code={git_commit()} device={device} backbone={cfg['backbone']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    records = [run_category(c, args, cfg, meta, device) for c in (args.categories or list(meta["config"]["categories"]))]
    write_summary(records, args.out)


if __name__ == "__main__":
    main()
