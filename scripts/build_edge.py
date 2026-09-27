"""Build + evaluate the edge (OpenVINO) variants of DefectLens.

  python scripts/build_edge.py --data /kaggle/working/data/processed --out /kaggle/working/edge

Variants = embedder {fp32, int8, int8mix} x memory-bank coreset ratio {10%, 1%, 0.1%}.
  * The embedder (shared by all categories) is exported in FP32 and INT8-quantised
    with NNCF twice: `int8` calibrated on normal TRAIN images only, `int8mix` on
    normal images + val SYNTHETIC defects. Calibrating activation ranges on normal
    data alone can clip exactly the out-of-distribution activations an anomaly
    detector relies on (measured: an INT8 kNN scorer calibrated on normals collapsed
    anomaly scores to normal level), so both are measured on the real test set.
  * The scorer (kNN + map) stays FP32 compute / FP16 weights for that reason.
  * Each variant's memory bank is re-built from ITS OWN embedder's features, so the
    bank and the query features always come from the same (quantised) network.
  * Scorers are saved with FP16 weights; accuracy is measured with the same
    FP16-rounded bank. Every saved scorer is checked against PyTorch on real images.
  * Thresholds: normal_quantile policy on val (same as train_patchcore.py).
Output: <out>/manifest.json (what EdgeInspector loads), <out>/report.md, <out>/samples/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import AnomalyDataset  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.data.validate import load_ready_dataset  # noqa: E402
from src.edge.export import quantize_int8, save, to_openvino  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.lineage import git_commit  # noqa: E402
from src.models.patchcore import BACKBONES, Embedder, Scorer, greedy_coreset  # noqa: E402


def load_split(data: Path, cat: str, split: str, size: int) -> dict:
    ds = AnomalyDataset(data, cat, split, build_eval_transform(size))
    items = [ds[i] for i in range(len(ds))]
    return {"x": [it["image"][None].numpy() for it in items],
            "labels": np.array([int(it["label"]) for it in items]),
            "masks": np.stack([it["mask"][0].numpy().astype(bool) for it in items]),
            "paths": [it["path"] for it in items]}


def embed_all(compiled: ov.CompiledModel, xs: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([compiled(x)[0] for x in xs]))  # (N, D, h, w)


@torch.no_grad()
def score_all(scorer: Scorer, emb: torch.Tensor, device) -> tuple[np.ndarray, np.ndarray]:
    s, a = zip(*(scorer(e[None].to(device)) for e in emb))
    return torch.cat(s).cpu().numpy(), torch.cat(a)[:, 0].cpu().numpy()


def check_exported(ov_scorer: ov.CompiledModel, scorer: Scorer, emb: torch.Tensor) -> float:
    """Max relative score difference OpenVINO vs PyTorch on a few real embeddings."""
    worst = 0.0
    for e in emb[:3]:
        ref = float(scorer.cpu()(e[None])[0])
        got = float(ov_scorer(e[None].numpy())[0][0])
        worst = max(worst, abs(ref - got) / max(abs(ref), 1e-6))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/processed"), type=Path)
    ap.add_argument("--out", default=Path("artifacts/edge"), type=Path)
    ap.add_argument("--config", default=ROOT / "configs/patchcore.yaml", type=Path)
    ap.add_argument("--ratios", nargs="*", type=float, default=[0.1, 0.01, 0.001])
    ap.add_argument("--save-all-max-ratio", type=float, default=0.01,
                    help="save scorers of every category for ratios <= this; above it only the first category "
                         "(10%% banks are ~50 MB each and only needed for the latency benchmark)")
    ap.add_argument("--calib-size", type=int, default=300)
    ap.add_argument("--calib-synthetic-frac", type=float, default=0.25,
                    help="share of val synthetic-defect images in the int8mix calibration set")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--backbone")
    ap.add_argument("--no-pretrained", action="store_true", help="tests only")
    args = ap.parse_args()

    meta = load_ready_dataset(args.data)
    cfg = yaml.safe_load(args.config.read_text())
    if args.backbone:
        cfg["backbone"] = args.backbone
    size = meta["config"]["image_size"]
    cats = args.categories or list(meta["config"]["categories"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    print(f"dataset run={meta['run_id']} code={git_commit()} backbone={cfg['backbone']} torch-device={dev}", flush=True)

    # ---- 1. embedder: FP32 IR + INT8 IR (calibrated on normal train images of all categories)
    torch.manual_seed(cfg["seed"])
    embedder = Embedder(cfg["backbone"], tuple(cfg["layers"]), not args.no_pretrained).eval()
    ov_fp32 = to_openvino(embedder, torch.zeros(1, 3, size, size))
    rng = np.random.default_rng(cfg["seed"])
    splits = {c: {s: load_split(args.data, c, s, size) for s in ("train", "val", "test")} for c in cats}
    normals = [x for c in cats for x in splits[c]["train"]["x"]]
    synthetic = [x for c in cats for x, y in zip(splits[c]["val"]["x"], splits[c]["val"]["labels"]) if y == 1]
    n_syn = int(args.calib_size * args.calib_synthetic_frac)
    calib_sets = {
        "int8": [normals[i] for i in rng.permutation(len(normals))[: args.calib_size]],
        "int8mix": [normals[i] for i in rng.permutation(len(normals))[: args.calib_size - n_syn]]
                   + [synthetic[i] for i in rng.permutation(len(synthetic))[:n_syn]],
    }
    embedders = {"fp32": {"xml": "embedder_fp32.xml",
                          "mb": save(ov_fp32, args.out / "embedder_fp32.xml", fp16_weights=False)}}
    for name, calib in calib_sets.items():
        t0 = time.perf_counter()
        q = quantize_int8(ov_fp32, calib, args.calib_size)
        embedders[name] = {"xml": f"embedder_{name}.xml", "mb": save(q, args.out / f"embedder_{name}.xml", False),
                           "calibration_images": len(calib), "quantize_seconds": round(time.perf_counter() - t0, 1)}
    core = ov.Core()

    # ---- 2. per precision: embed once, then build/evaluate a bank per coreset ratio
    variants: dict = {}
    for precision in embedders:
        compiled = core.compile_model(str(args.out / embedders[precision]["xml"]), "CPU")
        for cat in cats:
            t0 = time.perf_counter()
            emb = {s: embed_all(compiled, splits[cat][s]["x"]) for s in ("train", "val", "test")}
            embed_s = time.perf_counter() - t0
            d = emb["train"].shape[1]
            patches = emb["train"].permute(0, 2, 3, 1).reshape(-1, d).to(dev)
            for ratio in args.ratios:
                key = f"{precision}/r{ratio:g}"
                n = max(1, int(len(patches) * ratio))
                bank = patches[greedy_coreset(patches, n, seed=cfg["seed"])].half().float()  # as stored (FP16)
                scorer = Scorer(bank, cfg["sigma"], size).eval().to(dev)
                res = {}
                for s in ("val", "test"):
                    sc, am = score_all(scorer, emb[s], dev)
                    res[s] = {"scores": sc, "amaps": am, "labels": splits[cat][s]["labels"],
                              "masks": splits[cat][s]["masks"]}
                m = evaluate(res["val"], res["test"], cfg["threshold_policy"], cfg["target_fpr"])
                info = {"bank_size": n, "threshold": m["threshold"], "scorer": None,
                        "image_auroc": m["image_auroc"], "pixel_auroc": m["pixel_auroc"], "aupro_30": m["aupro_30"],
                        "f1": m["test_at_threshold"]["f1"], "fpr": m["test_at_threshold"]["fpr"],
                        "recall": m["test_at_threshold"]["recall"], "embed_seconds": round(embed_s, 1)}
                if ratio <= args.save_all_max_ratio or cat == cats[0]:
                    scorer_cpu = Scorer(bank.cpu(), cfg["sigma"], size).eval()
                    rel = Path(precision) / f"r{ratio:g}" / cat / "scorer.xml"
                    ov_scorer = to_openvino(scorer_cpu, emb["val"][:1])
                    info["scorer"] = rel.as_posix()
                    info["scorer_mb"] = save(ov_scorer, args.out / rel, fp16_weights=True)
                    info["export_rel_diff"] = check_exported(core.compile_model(ov_scorer, "CPU"), scorer_cpu, emb["val"])
                    if info["export_rel_diff"] > 1e-2:
                        raise RuntimeError(f"{key}/{cat}: exported scorer deviates {info['export_rel_diff']:.2e}")
                variants.setdefault(key, {"precision": precision, "coreset_ratio": ratio, "categories": {}})
                variants[key]["categories"][cat] = info
                print(f"[{key}] {cat}: img_AUROC={info['image_auroc']:.4f} AUPRO={info['aupro_30']:.4f} "
                      f"F1={info['f1']:.3f} FPR={info['fpr']:.3f} bank={n}", flush=True)

    # ---- 3. samples for the benchmark / demo (one normal + one defect per category)
    for cat in cats:
        test = splits[cat]["test"]
        for label in (0, 1):
            idx = int(np.flatnonzero(test["labels"] == label)[0])
            dst = args.out / "samples" / f"{cat}_{'good' if label == 0 else 'defect'}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.data / test["paths"][idx], dst)

    manifest = {"image_size": size, "backbone": cfg["backbone"], "layers": cfg["layers"], "sigma": cfg["sigma"],
                "threshold_policy": cfg["threshold_policy"], "target_fpr": cfg["target_fpr"],
                "embedders": embedders, "variants": variants,
                "lineage": {"git_commit": git_commit(), "dataset_run_id": meta["run_id"],
                            "openvino": ov.get_version(), "torch": torch.__version__}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report(manifest, args.out / "report.md")


def write_report(manifest: dict, path: Path) -> None:
    rows = ["| Embedder | Bank | Mean image AUROC | Mean AUPRO@0.3 | Mean F1 | Mean FPR | Banks total (MB, FP16) |",
            "|---|---|---|---|---|---|---|"]
    for key, v in manifest["variants"].items():
        cs = v["categories"].values()
        mean = lambda k: np.mean([c[k] for c in cs])  # noqa: E731
        bank_mb = sum(c["bank_size"] for c in cs) * _embed_dim(manifest) * 2 / 2**20  # FP16 = 2 bytes
        rows.append(f"| {v['precision']} ({manifest['embedders'][v['precision']]['mb']} MB) | "
                    f"{v['coreset_ratio']:.1%} | {mean('image_auroc'):.4f} | {mean('aupro_30'):.4f} | "
                    f"{mean('f1'):.3f} | {mean('fpr'):.3f} | {bank_mb:.1f} |")
    rows += ["", "Per category (image AUROC / F1):", "",
             "| Variant | " + " | ".join(next(iter(manifest["variants"].values()))["categories"]) + " |",
             "|---|" + "---|" * len(next(iter(manifest["variants"].values()))["categories"])]
    for key, v in manifest["variants"].items():
        rows.append(f"| {key} | " + " | ".join(f"{c['image_auroc']:.3f} / {c['f1']:.3f}"
                                               for c in v["categories"].values()) + " |")
    lin = manifest["lineage"]
    rows.append(f"\n_code `{lin['git_commit']}`, dataset run `{lin['dataset_run_id']}`, OpenVINO {lin['openvino']}_")
    path.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def _embed_dim(manifest: dict) -> int:
    channels = BACKBONES[manifest["backbone"]][1]
    return sum(channels[l] for l in manifest["layers"])


if __name__ == "__main__":
    main()
