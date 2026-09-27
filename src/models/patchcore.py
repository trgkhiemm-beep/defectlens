"""PatchCore (Roth et al., CVPR 2022, "Towards Total Recall in Industrial Anomaly Detection").

Implemented from the paper in plain PyTorch rather than through anomalib:
  * every op in `forward` is ONNX/OpenVINO friendly (conv, pooling, matmul, min),
    so the same module is exported unchanged in Step 3b;
  * no dependency on a fast-moving library API.

Pipeline:  image -> frozen CNN (layer2, layer3) -> 3x3 local avg pooling ->
           upsample + concat -> patch embeddings (B, D, h, w)
  fit:     embeddings of all normal train patches -> greedy k-center coreset = memory bank
  predict: patch score = distance to nearest memory-bank entry
           anomaly map  = upsampled + Gaussian-smoothed patch scores
           image score  = max of the anomaly map
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

# backbone name -> (torchvision weights enum, channels per layer)
BACKBONES = {
    "wide_resnet50_2": ("Wide_ResNet50_2_Weights", {"layer2": 512, "layer3": 1024}),
    "resnet18": ("ResNet18_Weights", {"layer2": 128, "layer3": 256}),
}


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = int(math.ceil(4 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1 = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    return (k1[:, None] * k1[None, :])[None, None]  # (1, 1, K, K)


@torch.no_grad()
def greedy_coreset(embeddings: torch.Tensor, n: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection (paper Alg. 1). Distances are computed in a random
    Johnson-Lindenstrauss projection to `proj_dim` dims for speed. Returns indices."""
    N, D = embeddings.shape
    if n >= N:
        return torch.arange(N, device=embeddings.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = embeddings
    if D > proj_dim:
        proj = torch.randn(D, proj_dim, generator=g).to(embeddings) / math.sqrt(proj_dim)
        z = embeddings @ proj
    start = int(torch.randint(N, (1,), generator=g))
    selected = torch.empty(n, dtype=torch.long, device=embeddings.device)
    selected[0] = start
    min_dist = torch.linalg.vector_norm(z - z[start], dim=1)
    for i in range(1, n):
        j = torch.argmax(min_dist)
        selected[i] = j
        min_dist = torch.minimum(min_dist, torch.linalg.vector_norm(z - z[j], dim=1))
    return selected


class PatchCore(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2", layers: tuple[str, ...] = ("layer2", "layer3"),
                 pretrained: bool = True, coreset_ratio: float = 0.1, sigma: float = 4.0):
        super().__init__()
        weights_name, channels = BACKBONES[backbone]
        weights = getattr(torchvision.models, weights_name).IMAGENET1K_V1 if pretrained else None
        cnn = getattr(torchvision.models, backbone)(weights=weights)
        self.extractor = create_feature_extractor(cnn, return_nodes={l: l for l in layers}).eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)
        self.layers = layers
        self.embed_dim = sum(channels[l] for l in layers)
        self.coreset_ratio = coreset_ratio
        self.register_buffer("blur_kernel", gaussian_kernel(sigma))
        self.register_buffer("memory_bank", torch.empty(0, self.embed_dim))
        self.config = dict(backbone=backbone, layers=list(layers), coreset_ratio=coreset_ratio, sigma=sigma)

    # ---------------------------------------------------------------- features
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) normalised images -> (B, D, h, w) locally-aware patch features."""
        feats = self.extractor(x)
        maps = [F.avg_pool2d(feats[l], kernel_size=3, stride=1, padding=1) for l in self.layers]
        size = maps[0].shape[-2:]
        maps = [maps[0]] + [F.interpolate(m, size=size, mode="bilinear", align_corners=False) for m in maps[1:]]
        return torch.cat(maps, dim=1)

    # --------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self, loader, device: torch.device, max_bank: int | None = None, seed: int = 0) -> dict:
        self.to(device).eval()
        chunks = []
        for batch in loader:
            e = self.embed(batch["image"].to(device))
            chunks.append(e.permute(0, 2, 3, 1).reshape(-1, self.embed_dim))
        all_emb = torch.cat(chunks)
        n = max(1, int(len(all_emb) * self.coreset_ratio))
        if max_bank:
            n = min(n, max_bank)
        idx = greedy_coreset(all_emb, n, seed=seed)
        self.memory_bank = all_emb[idx].contiguous()
        return {"n_patches": len(all_emb), "bank_size": n, "bank_mb": round(self.memory_bank.numel() * 4 / 2**20, 1)}

    # ----------------------------------------------------------------- predict
    def nearest_distance(self, emb: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """(P, D) -> (P,) Euclidean distance to the nearest memory-bank entry.
        Uses ||a||^2 - 2ab + ||b||^2 (a matmul) instead of cdist: exportable and fast."""
        bank = self.memory_bank
        bank_sq = (bank * bank).sum(1)
        out = []
        for a in emb.split(chunk):
            d2 = (a * a).sum(1, keepdim=True) - 2 * a @ bank.T + bank_sq[None]
            out.append(d2.min(dim=1).values.clamp_min(0).sqrt())
        return torch.cat(out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) -> image scores (B,), anomaly maps (B, 1, H, W)."""
        if self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore is not fitted")
        e = self.embed(x)
        b, d, h, w = e.shape
        scores = self.nearest_distance(e.permute(0, 2, 3, 1).reshape(-1, d)).reshape(b, 1, h, w)
        amap = F.interpolate(scores, size=x.shape[-2:], mode="bilinear", align_corners=False)
        pad = self.blur_kernel.shape[-1] // 2
        amap = F.conv2d(F.pad(amap, (pad, pad, pad, pad), mode="reflect"), self.blur_kernel)
        return amap.amax(dim=(1, 2, 3)), amap

    # --------------------------------------------------------------- persist
    def state(self) -> dict:
        return {"config": self.config, "memory_bank": self.memory_bank.cpu()}

    @classmethod
    def from_state(cls, state: dict, pretrained: bool = True) -> "PatchCore":
        c = state["config"]
        m = cls(c["backbone"], tuple(c["layers"]), pretrained, c["coreset_ratio"], c["sigma"])
        m.memory_bank = state["memory_bank"]
        return m.eval()
