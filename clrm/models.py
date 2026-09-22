from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn


TensorDict = Mapping[int, torch.Tensor]


def masked_moments(content: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return concatenated masked mean/std over the token dimension.

    Args:
        content: [batch, seq, dim]
        mask: [batch, seq] boolean tensor
    """
    if content.ndim != 3 or mask.ndim != 2:
        raise ValueError("content must be [B,T,D] and mask must be [B,T]")
    if content.shape[:2] != mask.shape:
        raise ValueError("content/mask shape mismatch")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every candidate must contain at least one valid response token")

    w = mask.to(dtype=content.dtype)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (w.unsqueeze(-1) * content).sum(dim=1) / denom
    var = (w.unsqueeze(-1) * (content - mean.unsqueeze(1)).square()).sum(dim=1) / denom
    return torch.cat([mean, torch.sqrt(var + eps)], dim=-1)


def _check_depths(hs: TensorDict, depths: Sequence[int]) -> None:
    missing = [d for d in depths if d not in hs]
    if missing:
        raise KeyError(f"missing hidden-state depths: {missing}")


class SharedProjector(nn.Module):
    def __init__(self, hidden_size: int, projection_size: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_size, projection_size, bias=False)
        self.norm = nn.LayerNorm(projection_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class MDState(nn.Module):
    """Multi-Depth State Set verifier used as the paper's primary method."""

    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        projection_size: int = 128,
        token_feature_size: int = 32,
        readout_size: int = 96,
    ):
        super().__init__()
        if len(depths) < 1:
            raise ValueError("depths must be non-empty")
        self.depths = tuple(int(d) for d in depths)
        self.projector = SharedProjector(hidden_size, projection_size)
        self.phi = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        pooled_size = len(self.depths) * 2 * token_feature_size
        self.readout = nn.Sequential(
            nn.Linear(pooled_size, readout_size),
            nn.ReLU(),
            nn.Linear(readout_size, 1),
        )

    def projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_depths(hs, self.depths)
        blocks = [masked_moments(self.phi(self.projected(hs[d])), mask) for d in self.depths]
        feature = torch.cat(blocks, dim=-1)
        return {"score": self.readout(feature).squeeze(-1), "feature": feature}


class SingleState(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depth: int,
        projection_size: int = 128,
        token_feature_size: int = 32,
        readout_size: int = 96,
    ):
        super().__init__()
        self.depth = int(depth)
        self.projector = SharedProjector(hidden_size, projection_size)
        self.phi = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        self.readout = nn.Sequential(
            nn.Linear(2 * token_feature_size, readout_size),
            nn.ReLU(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_depths(hs, (self.depth,))
        feature = masked_moments(self.phi(self.projector(hs[self.depth])), mask)
        return {"score": self.readout(feature).squeeze(-1), "feature": feature}


class MultiEndpoint(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        projection_size: int = 128,
        readout_size: int = 96,
    ):
        super().__init__()
        self.depths = tuple(int(d) for d in depths)
        self.projector = SharedProjector(hidden_size, projection_size)
        self.readout = nn.Sequential(
            nn.Linear(len(self.depths) * projection_size, readout_size),
            nn.ReLU(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_depths(hs, self.depths)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("empty response sequence")
        last = mask.shape[1] - 1 - torch.flip(mask, dims=[1]).to(torch.float32).argmax(dim=1)
        rows = torch.arange(mask.shape[0], device=mask.device)
        feature = torch.cat([self.projector(hs[d][rows, last]) for d in self.depths], dim=-1)
        return {"score": self.readout(feature).squeeze(-1), "feature": feature}


class MultiDiff(nn.Module):
    """Token-axis adjacent differences pooled independently at multiple depths."""

    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        projection_size: int = 128,
        token_feature_size: int = 32,
        readout_size: int = 96,
    ):
        super().__init__()
        self.depths = tuple(int(d) for d in depths)
        self.projector = SharedProjector(hidden_size, projection_size)
        self.phi = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        pooled_size = len(self.depths) * 2 * token_feature_size
        self.readout = nn.Sequential(
            nn.Linear(pooled_size, readout_size),
            nn.ReLU(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_depths(hs, self.depths)
        transition_mask = mask[:, 1:] & mask[:, :-1]
        if not bool(transition_mask.any(dim=1).all()):
            raise ValueError("MultiDiff requires at least two valid response tokens per candidate")
        blocks = []
        for d in self.depths:
            z = self.projector(hs[d])
            delta = z[:, 1:] - z[:, :-1]
            blocks.append(masked_moments(self.phi(delta), transition_mask))
        feature = torch.cat(blocks, dim=-1)
        return {"score": self.readout(feature).squeeze(-1), "feature": feature}


class ResidualOnly(nn.Module):
    """Matched-token depth transformation verifier."""

    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        projection_size: int = 128,
        token_feature_size: int = 32,
        readout_size: int = 96,
    ):
        super().__init__()
        self.depths = tuple(int(d) for d in depths)
        if len(self.depths) != 3:
            raise ValueError("ResidualOnly expects exactly three depths: early, middle, final")
        self.projector = SharedProjector(hidden_size, projection_size)
        self.phi = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        self.readout = nn.Sequential(
            nn.Linear(4 * token_feature_size, readout_size),
            nn.ReLU(),
            nn.Linear(readout_size, 1),
        )

    def projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)

    def residual_features(self, hs: TensorDict, mask: torch.Tensor) -> torch.Tensor:
        _check_depths(hs, self.depths)
        e, m, f = self.depths
        z = {d: self.projected(hs[d]) for d in self.depths}
        r1 = masked_moments(self.phi(z[m] - z[e]), mask)
        r2 = masked_moments(self.phi(z[f] - z[m]), mask)
        return torch.cat([r1, r2], dim=-1)

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.residual_features(hs, mask)
        return {"score": self.readout(feature).squeeze(-1), "feature": feature}


class StateResidualConcat(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        projection_size: int = 128,
        token_feature_size: int = 32,
        readout_size: int = 96,
    ):
        super().__init__()
        self.depths = tuple(int(d) for d in depths)
        if len(self.depths) != 3:
            raise ValueError("StateResidualConcat expects exactly three depths")
        self.projector = SharedProjector(hidden_size, projection_size)
        self.phi_state = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        self.phi_residual = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        state_size = len(self.depths) * 2 * token_feature_size
        residual_size = 4 * token_feature_size
        self.readout = nn.Sequential(
            nn.Linear(state_size + residual_size, readout_size),
            nn.ReLU(),
            nn.Linear(readout_size, 1),
        )

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_depths(hs, self.depths)
        e, m, f = self.depths
        z = {d: self.projector(hs[d]) for d in self.depths}
        state = torch.cat([masked_moments(self.phi_state(z[d]), mask) for d in self.depths], dim=-1)
        residual = torch.cat(
            [
                masked_moments(self.phi_residual(z[m] - z[e]), mask),
                masked_moments(self.phi_residual(z[f] - z[m]), mask),
            ],
            dim=-1,
        )
        feature = torch.cat([state, residual], dim=-1)
        return {
            "score": self.readout(feature).squeeze(-1),
            "feature": feature,
            "state_feature": state,
            "residual_feature": residual,
        }


class StateResidualWeighted(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        projection_size: int = 128,
        token_feature_size: int = 32,
        latent_size: int = 96,
    ):
        super().__init__()
        self.depths = tuple(int(d) for d in depths)
        if len(self.depths) != 3:
            raise ValueError("StateResidualWeighted expects exactly three depths")
        self.projector = SharedProjector(hidden_size, projection_size)
        self.phi_state = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        self.phi_residual = nn.Sequential(nn.Linear(projection_size, token_feature_size), nn.ReLU())
        self.state_adapter = nn.Sequential(nn.Linear(6 * token_feature_size, latent_size), nn.ReLU())
        self.residual_adapter = nn.Sequential(nn.Linear(4 * token_feature_size, latent_size), nn.ReLU())
        self.fusion_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.score_head = nn.Linear(latent_size, 1)

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits, dim=0)

    def forward(self, hs: TensorDict, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _check_depths(hs, self.depths)
        e, m, f = self.depths
        z = {d: self.projector(hs[d]) for d in self.depths}
        state = torch.cat([masked_moments(self.phi_state(z[d]), mask) for d in self.depths], dim=-1)
        residual = torch.cat(
            [
                masked_moments(self.phi_residual(z[m] - z[e]), mask),
                masked_moments(self.phi_residual(z[f] - z[m]), mask),
            ],
            dim=-1,
        )
        u_state = self.state_adapter(state)
        u_residual = self.residual_adapter(residual)
        weights = self.fusion_weights()
        fused = weights[0] * u_state + weights[1] * u_residual
        return {
            "score": self.score_head(fused).squeeze(-1),
            "state_feature": state,
            "residual_feature": residual,
            "state_latent": u_state,
            "residual_latent": u_residual,
            "alpha_state": weights[0],
            "beta_residual": weights[1],
        }


MODEL_REGISTRY = {
    "md_state": MDState,
    "single_state": SingleState,
    "multi_endpoint": MultiEndpoint,
    "multi_diff": MultiDiff,
    "residual_only": ResidualOnly,
    "concat": StateResidualConcat,
    "weighted": StateResidualWeighted,
}


def build_model(name: str, hidden_size: int, depths: Sequence[int]) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'. available: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    if cls is SingleState:
        return cls(hidden_size=hidden_size, depth=int(depths[-1]))
    return cls(hidden_size=hidden_size, depths=depths)
