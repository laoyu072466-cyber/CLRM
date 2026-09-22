import torch

from clrm.models import (
    MDState,
    MultiDiff,
    MultiEndpoint,
    ResidualOnly,
    SingleState,
    StateResidualConcat,
    StateResidualWeighted,
    masked_moments,
)


def sample():
    torch.manual_seed(0)
    hs = {d: torch.randn(4, 7, 16) for d in (2, 4, 6)}
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    return hs, mask


def test_masked_moments_shape():
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert masked_moments(x, mask).shape == (3, 16)


def test_all_models_forward():
    hs, mask = sample()
    models = [
        MDState(16, [2, 4, 6]),
        SingleState(16, 6),
        MultiEndpoint(16, [2, 4, 6]),
        MultiDiff(16, [2, 4, 6]),
        ResidualOnly(16, [2, 4, 6]),
        StateResidualConcat(16, [2, 4, 6]),
        StateResidualWeighted(16, [2, 4, 6]),
    ]
    for model in models:
        out = model(hs, mask)
        assert out["score"].shape == (4,)
        assert torch.isfinite(out["score"]).all()
