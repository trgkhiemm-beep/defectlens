"""Pick ONE edge variant and copy only what serving needs into a small bundle.

  python scripts/export_bundle.py --edge artifacts/edge --precision int8 --bank r0.01 --out deploy/model

Bundle = shared embedder + one scorer per category + samples + trimmed manifest
(with a `default` entry that EdgeInspector.from_bundle loads) + MODEL_CARD.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", default=Path("artifacts/edge"), type=Path)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--bank", default="r0.01")
    ap.add_argument("--out", default=Path("deploy/model"), type=Path)
    args = ap.parse_args()

    m = json.loads((args.edge / "manifest.json").read_text())
    key = f"{args.precision}/{args.bank}"
    variant = m["variants"][key]
    missing = [c for c, i in variant["categories"].items() if not i.get("scorer")]
    if missing:
        raise SystemExit(f"{key} has no exported scorer for {missing}; pick a smaller bank")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    emb = m["embedders"][args.precision]
    files = [emb["xml"]] + [i["scorer"] for i in variant["categories"].values()]
    for f in files:
        for suffix in (".xml", ".bin"):
            src = (args.edge / f).with_suffix(suffix)
            dst = (args.out / f).with_suffix(suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    shutil.copytree(args.edge / "samples", args.out / "samples")

    bundle = {k: m[k] for k in ("image_size", "backbone", "layers", "sigma", "threshold_policy", "target_fpr", "lineage")}
    bundle.update(default={"precision": args.precision, "bank": args.bank},
                  embedders={args.precision: emb}, variants={key: variant})
    (args.out / "manifest.json").write_text(json.dumps(bundle, indent=2))

    rows = ["| Category | Image AUROC | AUPRO@0.3 | F1 | FPR | Threshold |", "|---|---|---|---|---|---|"]
    for c, i in variant["categories"].items():
        rows.append(f"| {c} | {i['image_auroc']:.3f} | {i['aupro_30']:.3f} | {i['f1']:.3f} | {i['fpr']:.3f} | "
                    f"{i['threshold']:.3f} |")
    size_mb = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 2**20
    card = [f"# DefectLens model bundle `{key}`", "",
            f"PatchCore ({m['backbone']} {'+'.join(m['layers'])}), {args.precision.upper()} embedder, "
            f"coreset bank {variant['coreset_ratio']:.1%}, OpenVINO IR. Bundle size {size_mb:.1f} MB.",
            f"Threshold policy `{m['threshold_policy']}` (target FPR {m['target_fpr']}) chosen on validation data.", "",
            *rows, "", f"Lineage: {json.dumps(m['lineage'])}", "",
            "Trained on MVTec AD (CC BY-NC-SA 4.0): non-commercial use only."]
    (args.out / "MODEL_CARD.md").write_text("\n".join(card) + "\n")
    print(f"bundle {key} -> {args.out} ({size_mb:.1f} MB, {len(variant['categories'])} categories)")


if __name__ == "__main__":
    main()
