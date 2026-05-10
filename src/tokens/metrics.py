from __future__ import annotations

# Alignment metrics adapted from https://github.com/minyoungg/platonic-rep.

import numpy as np
import torch
import torch.nn.functional as F


def remove_outliers(feats: torch.Tensor, q: float) -> torch.Tensor:
    if q == 1:
        return feats
    q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    return feats.clamp(-q_val, q_val)


def nearest_neighbors(feats: torch.Tensor, k: int) -> torch.Tensor:
    if feats.ndim != 2:
        raise ValueError(f"expected 2D features, got {tuple(feats.shape)}")
    k = min(k, feats.shape[0] - 1)
    sims = feats @ feats.T
    sims.fill_diagonal_(-torch.inf)
    return sims.argsort(dim=1, descending=True)[:, :k]


def mutual_knn(feats_a: torch.Tensor, feats_b: torch.Tensor, k: int) -> float:
    knn_a = nearest_neighbors(feats_a, k)
    knn_b = nearest_neighbors(feats_b, k)
    n, k = knn_a.shape
    rows = torch.arange(n, device=feats_a.device).unsqueeze(1)
    mask_a = torch.zeros(n, n, device=feats_a.device)
    mask_b = torch.zeros(n, n, device=feats_a.device)
    mask_a[rows, knn_a] = 1.0
    mask_b[rows, knn_b] = 1.0
    return float(((mask_a * mask_b).sum(dim=1) / k).mean().item())


def hsic_debiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    m = K.shape[0]
    if m < 4:
        raise ValueError("debiased HSIC requires at least 4 samples")
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    hsic = (
        torch.sum(K_tilde * L_tilde.T)
        + torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2))
        - 2 * torch.sum(K_tilde @ L_tilde) / (m - 2)
    )
    return hsic / (m * (m - 3))


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    return K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    K_centered = center_kernel(K)
    L_centered = center_kernel(L)
    return torch.sum(K_centered * L_centered)


def linear_cka(feats_a: torch.Tensor, feats_b: torch.Tensor) -> float:
    K = feats_a @ feats_a.T
    L = feats_b @ feats_b.T
    hsic_kk = hsic_biased(K, K)
    hsic_ll = hsic_biased(L, L)
    hsic_kl = hsic_biased(K, L)
    return float((hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item())


def linear_debiased_cka(feats_a: torch.Tensor, feats_b: torch.Tensor) -> float:
    K = feats_a @ feats_a.T
    L = feats_b @ feats_b.T
    hsic_kk = hsic_debiased(K, K)
    hsic_ll = hsic_debiased(L, L)
    hsic_kl = hsic_debiased(K, L)
    return float((hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item())


def normalized_pair(
    XA: np.ndarray,
    XB: np.ndarray,
    *,
    clip_q: float = 0.95,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    XA_t = torch.as_tensor(XA, device=device, dtype=torch.float32)
    XB_t = torch.as_tensor(XB, device=device, dtype=torch.float32)
    if XA_t.ndim != 2 or XB_t.ndim != 2:
        raise ValueError(f"expected 2D arrays, got {tuple(XA_t.shape)} and {tuple(XB_t.shape)}")
    if XA_t.shape[0] != XB_t.shape[0]:
        raise ValueError(f"sample counts differ: {XA_t.shape[0]} vs {XB_t.shape[0]}")
    if XA_t.shape[0] < 2:
        raise ValueError("alignment requires at least 2 paired samples")

    XA_t = F.normalize(remove_outliers(XA_t, clip_q), p=2, dim=-1)
    XB_t = F.normalize(remove_outliers(XB_t, clip_q), p=2, dim=-1)
    return XA_t, XB_t


def compute_metric_values(
    XA: np.ndarray,
    XB: np.ndarray,
    *,
    metrics: tuple[str, ...] = ("debiased_cka",),
    ks: tuple[int, ...] = (),
    clip_q: float = 0.95,
    device: str | None = None,
) -> dict[str, float | dict[int, float]]:
    XA_t, XB_t = normalized_pair(XA, XB, clip_q=clip_q, device=device)

    results: dict[str, float | dict[int, float]] = {}
    if "cka" in metrics:
        results["cka"] = linear_cka(XA_t, XB_t)
    if "debiased_cka" in metrics:
        results["debiased_cka"] = linear_debiased_cka(XA_t, XB_t)
    if "mknn" in metrics:
        if not ks:
            raise ValueError("mKNN requires at least one k in --ks")
        results["mknn"] = {k: mutual_knn(XA_t, XB_t, k) for k in ks}
    return results


def compute_alignment(
    XA: np.ndarray,
    XB: np.ndarray,
    ks: tuple[int, ...] = (10,),
    clip_q: float = 0.95,
    device: str | None = None,
) -> dict[int, dict[str, float]]:
    values = compute_metric_values(
        XA,
        XB,
        metrics=("cka", "mknn"),
        ks=ks,
        clip_q=clip_q,
        device=device,
    )
    mknn_values = values["mknn"]
    assert isinstance(mknn_values, dict)
    cka_value = values["cka"]
    assert isinstance(cka_value, float)
    return {
        k: {
            "mknn": mknn_values[k],
            "cka": cka_value,
        }
        for k in ks
    }
