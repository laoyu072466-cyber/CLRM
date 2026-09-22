import numpy as np

from clrm.eval import best_at_k, pair_macro


def metadata():
    return [
        {"problem_id": "a", "candidate_index": 0, "label": 1},
        {"problem_id": "a", "candidate_index": 1, "label": 0},
        {"problem_id": "b", "candidate_index": 0, "label": 1},
        {"problem_id": "b", "candidate_index": 1, "label": 0},
    ]


def test_pair_macro_perfect():
    macro, micro = pair_macro(metadata(), np.array([2.0, 1.0, 3.0, 0.0]))
    assert macro == 1.0
    assert micro == 1.0


def test_best_at_k():
    subsets = {"a": [[0, 1]], "b": [[0, 1]]}
    value = best_at_k(metadata(), np.array([2.0, 1.0, 3.0, 0.0]), subsets)
    assert value == 1.0
