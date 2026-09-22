import numpy as np

from analysis.mechanism import geometry


def test_geometry_shapes():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 8))
    b = rng.normal(size=(5, 8))
    c = rng.normal(size=(5, 8))
    m1, m2, cos = geometry(a, b, c)
    assert m1.shape == (5,)
    assert m2.shape == (5,)
    assert cos.shape == (5,)
    assert np.all(cos <= 1.0) and np.all(cos >= -1.0)
