"""Assemble the Hugging Face Space build context in dist/space (Dockerfile + serving code + model bundle).

  python scripts/export_bundle.py            # -> deploy/model
  python scripts/make_space.py               # -> dist/space
  huggingface-cli upload <hf-user>/defectlens dist/space . --repo-type=space
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", ".dockerignore", "requirements-serve.txt", "src/__init__.py",
         "src/edge/__init__.py", "src/edge/preprocess.py", "src/edge/runtime.py", "app/__init__.py", "app/api.py", "app/main.py",
         "app/service.py", "app/ui.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ROOT / "deploy/model", type=Path)
    ap.add_argument("--out", default=ROOT / "dist/space", type=Path)
    args = ap.parse_args()
    if not (args.model / "manifest.json").exists():
        raise SystemExit(f"{args.model} has no bundle: run scripts/export_bundle.py first")
    if args.out.exists():
        shutil.rmtree(args.out)
    for f in FILES:
        (args.out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, args.out / f)
    shutil.copytree(args.model, args.out / "deploy/model")
    shutil.copy(ROOT / "deploy/space/README.md", args.out / "README.md")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 2**20
    print(f"Space context -> {args.out} ({size:.1f} MB). Upload with:\n"
          f"  huggingface-cli upload <hf-user>/defectlens {args.out} . --repo-type=space")


if __name__ == "__main__":
    main()
