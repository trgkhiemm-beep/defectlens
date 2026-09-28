"""Export the in-browser build (ONNX Runtime Web) of one edge variant into web/models.

  python scripts/export_web.py --edge artifacts/edge --variant fp32/r0.01

Why FP32 (not INT8) for the browser: the INT8 embedder is an OpenVINO/NNCF graph and the
memory bank must come from the SAME network as the query features. The fp32 variant's
bank was built from FP32 features, so an FP32 ONNX embedder reproduces it exactly.
Weights are STORED as FP16 (+ Cast to FP32 at load), halving the download while compute
stays FP32; the scorer banks were already FP16 on disk.

Every exported pair is checked against the OpenVINO pipeline on the sample images
(score within 1%, identical verdict) before anything is written as "ready".
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import openvino as ov
import torch
import yaml
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.edge.preprocess import MEAN, STD, preprocess  # noqa: E402
from src.models.patchcore import Embedder, Scorer  # noqa: E402

OPSET = 17


def export(module: torch.nn.Module, example: torch.Tensor, path: Path, outputs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module.eval(), (example,), str(path), input_names=["input"], output_names=outputs,
                          opset_version=OPSET, dynamo=False)


def store_weights_fp16(path: Path, min_elements: int = 1024) -> None:
    """Large FP32 initializers -> FP16 initializer + Cast(to FP32). Compute stays FP32 (every
    ONNX Runtime Web backend supports it), file size halves."""
    m = onnx.load(str(path))
    g = m.graph
    casts = []
    for init in list(g.initializer):
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims) < min_elements:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        name16 = init.name + "_fp16"
        g.initializer.remove(init)
        g.initializer.append(numpy_helper.from_array(arr, name16))
        casts.append(helper.make_node("Cast", [name16], [init.name], to=onnx.TensorProto.FLOAT,
                                      name=init.name + "_cast"))
    for i, node in enumerate(casts):
        g.node.insert(i, node)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def bank_from_ir(xml: Path) -> np.ndarray:
    model = ov.Core().read_model(str(xml))
    for op in model.get_ops():
        if op.get_type_name() == "Constant" and op.get_friendly_name().startswith("self.bank"):
            return op.get_data().astype(np.float32)
    raise ValueError(f"no memory bank constant in {xml}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default=ROOT / "artifacts/edge", type=Path)
    ap.add_argument("--variant", default="fp32/r0.01")
    ap.add_argument("--out", default=ROOT / "web", type=Path)
    ap.add_argument("--no-pretrained", action="store_true", help="tests only")
    args = ap.parse_args()

    m = json.loads((args.edge / "manifest.json").read_text())
    variant = m["variants"][args.variant]
    precision = variant["precision"]
    if precision != "fp32":
        raise SystemExit("browser export needs an fp32 variant (bank must match the FP32 ONNX embedder)")
    size = m["image_size"]
    models = args.out / "models"
    if models.exists():
        shutil.rmtree(models)

    seed = yaml.safe_load((ROOT / "configs/patchcore.yaml").read_text())["seed"]
    torch.manual_seed(seed)  # same init as build_edge.py (only matters for random-weight test backbones)
    embedder = Embedder(m["backbone"], tuple(m["layers"]), not args.no_pretrained).eval()
    export(embedder, torch.zeros(1, 3, size, size), models / "embedder.onnx", ["embedding"])
    store_weights_fp16(models / "embedder.onnx")

    categories = {}
    for cat, info in variant["categories"].items():
        bank = bank_from_ir(args.edge / info["scorer"])
        with torch.no_grad():
            emb = embedder(torch.zeros(1, 3, size, size))
        export(Scorer(torch.from_numpy(bank), m["sigma"], size), emb, models / f"{cat}.onnx", ["score", "anomaly_map"])
        store_weights_fp16(models / f"{cat}.onnx")
        categories[cat] = {"model": f"models/{cat}.onnx", "threshold": info["threshold"], "bank_size": len(bank),
                           "image_auroc": info["image_auroc"], "aupro_30": info["aupro_30"], "f1": info["f1"],
                           "fpr": info["fpr"], "mb": round((models / f"{cat}.onnx").stat().st_size / 2**20, 2)}

    # ---- parity check vs the OpenVINO pipeline on the bundled samples
    from src.edge.runtime import EdgeInspector
    insp = EdgeInspector(args.edge, precision, args.variant.split("/")[1], "CPU")
    emb_sess = ort.InferenceSession(str(models / "embedder.onnx"), providers=["CPUExecutionProvider"])
    worst = 0.0
    for cat in categories:
        sc_sess = ort.InferenceSession(str(models / f"{cat}.onnx"), providers=["CPUExecutionProvider"])
        for kind in ("good", "defect"):
            img = cv2.imread(str(args.edge / "samples" / f"{cat}_{kind}.png"))
            e = emb_sess.run(None, {"input": preprocess(img, size)})[0]
            s = float(sc_sess.run(None, {"input": e})[0][0])
            ref = insp.predict(img, cat)
            rel = abs(s - ref["score"]) / ref["score"]
            worst = max(worst, rel)
            same = (s >= ref["threshold"]) == ref["is_defect"]
            print(f"{cat:10s} {kind:6s} onnx={s:.4f} openvino={ref['score']:.4f} rel={rel:.2%} verdict_match={same}")
            if rel > 0.01 or not same:
                raise SystemExit("ONNX export deviates from the OpenVINO pipeline")

    shutil.copytree(args.edge / "samples", args.out / "samples", dirs_exist_ok=True)
    manifest = {"image_size": size, "mean": MEAN.tolist(), "std": STD.tolist(), "variant": args.variant,
                "backbone": m["backbone"], "layers": m["layers"], "threshold_policy": m["threshold_policy"],
                "target_fpr": m["target_fpr"], "embedder": {"model": "models/embedder.onnx",
                "mb": round((models / "embedder.onnx").stat().st_size / 2**20, 2)},
                "categories": categories, "parity_max_rel_diff": worst, "lineage": m["lineage"]}
    (models / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in models.glob("*.onnx")) / 2**20
    print(f"web models -> {models} ({total:.1f} MB), parity max rel diff {worst:.3%}")


if __name__ == "__main__":
    main()
