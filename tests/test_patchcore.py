"""PatchCore + metrics tests (CPU, random-weight resnet18: no downloads)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.validate import load_ready_dataset  # noqa: E402
from src.eval.metrics import aupro, auroc, best_f1_threshold, evaluate  # noqa: E402
from src.models.patchcore import PatchCore, greedy_coreset  # noqa: E402


# ---------------------------------------------------------------- metrics
def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=300), 1)  # rounding creates ties
    y = rng.integers(0, 2, 300)
    pos, neg = s[y == 1], s[y == 0]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert auroc(s, y) == pytest.approx(brute, abs=1e-12)


def test_aupro_perfect_and_inverted_maps():
    masks = np.zeros((4, 32, 32), bool)
    masks[0, 5:10, 5:10] = masks[1, 20:30, 2:6] = True
    rng = np.random.default_rng(1)
    perfect = masks + rng.uniform(0, 0.1, masks.shape)
    assert aupro(perfect, masks) > 0.99
    assert aupro(-perfect, masks) < 0.01


def test_threshold_selection_and_oracle_gap():
    val = {"scores": np.array([0.1, 0.2, 0.8, 0.9]), "labels": np.array([0, 0, 1, 1])}
    assert best_f1_threshold(val["scores"], val["labels"])["f1"] == 1.0
    test = {"scores": np.array([0.1, 0.6, 0.7, 0.95]), "labels": np.array([0, 0, 1, 1]),
            "amaps": np.zeros((4, 8, 8)), "masks": np.zeros((4, 8, 8), bool)}
    test["masks"][2:, 2:4, 2:4] = True
    test["amaps"][2:, 2:4, 2:4] = 1
    m = evaluate({**val, "amaps": test["amaps"], "masks": test["masks"]}, test)
    assert m["test_oracle"]["f1"] == 1.0
    assert m["f1_gap_vs_oracle"] >= 0


# -------------------------------------------------------------- patchcore
def test_coreset_is_unique_and_keeps_outliers():
    emb = torch.randn(500, 16)
    emb[123] = 50.0  # far outlier must be selected by k-center
    idx = greedy_coreset(emb, 20)
    assert len(set(idx.tolist())) == 20 and 123 in idx.tolist()


def _model(seed=0):
    torch.manual_seed(seed)
    return PatchCore("resnet18", pretrained=False, coreset_ratio=0.5)


def test_nearest_distance_equals_cdist():
    m = _model()
    m.memory_bank = torch.randn(300, m.embed_dim)
    q = torch.randn(50, m.embed_dim)
    ref = torch.cdist(q, m.memory_bank).min(dim=1).values
    assert torch.allclose(m.nearest_distance(q, chunk=16), ref, atol=1e-3)


def _images(n, rng, defect=False):
    x = 0.3 + 0.02 * rng.standard_normal((n, 3, 64, 64)).astype(np.float32)
    if defect:
        x[:, :, 40:52, 8:20] = 3.0  # bright square
    return torch.from_numpy(x)


def test_anomaly_map_localises_defect_and_state_roundtrip():
    rng = np.random.default_rng(0)
    m = _model()
    stats = m.fit([{"image": _images(8, rng)}], torch.device("cpu"))
    assert stats["bank_size"] == stats["n_patches"] // 2
    s_ok, _ = m(_images(2, rng))
    x_bad = _images(2, rng, defect=True)
    s_bad, amap = m(x_bad)
    assert amap.shape == (2, 1, 64, 64) and (s_bad > s_ok.max()).all()
    y, x = np.unravel_index(amap[0, 0].argmax().item(), (64, 64))
    assert 32 <= y <= 60 and 0 <= x <= 28, (y, x)

    torch.manual_seed(0)  # same random backbone
    m2 = PatchCore.from_state(m.state(), pretrained=False)
    assert torch.allclose(m2(x_bad)[0], s_bad, atol=1e-4)


# ------------------------------------------------------------- end to end
def test_train_script_end_to_end(built, tmp_path):
    data = built[0]
    cfg = yaml.safe_load((ROOT / "configs/patchcore.yaml").read_text())
    cfg.update(num_workers=0, batch_size=8, latency_runs=2)
    cfg_path = tmp_path / "pc.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "art"
    subprocess.run([sys.executable, "scripts/train_patchcore.py", "--data", str(data), "--out", str(out),
                    "--config", str(cfg_path), "--backbone", "resnet18", "--no-pretrained", "--device", "cpu",
                    "--categories", "transistor"], cwd=ROOT, check=True)
    rec = json.loads((out / "transistor" / "metrics.json").read_text())
    m = rec["metrics"]
    for k in ("image_auroc", "pixel_auroc", "aupro_30"):
        assert 0.0 <= m[k] <= 1.0, k
    assert set(m["per_defect_auroc"]) == {"scratch"}
    assert rec["lineage"]["dataset_run_id"] == json.loads((data / "dataset_meta.json").read_text())["run_id"]
    assert (out / "transistor" / "model.pt").exists() and (out / "transistor" / "examples.png").exists()
    assert "| transistor |" in (out / "summary.md").read_text()


def test_training_refuses_unvalidated_dataset(built, tmp_path):
    data = tmp_path / "copy"
    shutil.copytree(built[0], data)
    meta = json.loads((data / "dataset_meta.json").read_text())
    meta["validation"]["passed"] = False
    (data / "dataset_meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="did not pass validation"):
        load_ready_dataset(data)
