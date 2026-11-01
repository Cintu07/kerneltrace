import numpy as np
import pytest

from kerneltrace.metrics import stats as S


def test_overlap():
    assert S.topk_overlap([1, 5, 2], [1, 5, 2], 2) == 1
    assert S.topk_overlap([1, 5, 2], [5, 1, 2], 1) == 0
    assert S.topk_overlap([], [], 2) is None


def test_restoration():
    assert S.relative_restoration(2, 0, 1) == 0.5
    assert S.relative_restoration(0, 0, 0) is None
    assert S.relative_restoration(2, 0, 0) == 0


def test_restoration_array_refuses_zero_contrast():
    with pytest.raises(ValueError):
        S.restoration_array(np.array([1.0]), np.array([1.0]), np.array([0.5]))


def test_near_tie_flagged():
    out = S.topk_with_ties([3.0, 1.0, 1.0 + 1e-9], [3.0, 1.0 + 1e-9, 1.0], k=2, tie_tol=1e-6)
    assert out["near_tie"]
    assert not S.topk_with_ties([3.0, 2.0, 1.0], [3.0, 2.0, 1.0], k=2)["near_tie"]


def test_wilson_contains_point_estimate():
    lo, hi = S.wilson(3, 40)
    assert lo < 3 / 40 < hi
    assert S.wilson(0, 0) == (0.0, 1.0)
    lo, hi = S.wilson(0, 50)
    assert lo == 0.0 and hi < 0.1


def test_cluster_bootstrap_resamples_whole_clusters():
    # two clusters with constant values, so every resample mean is some mix of 0 and 10.
    values = np.array([0.0] * 5 + [10.0] * 5)
    clusters = np.array([0] * 5 + [1] * 5)
    boot = S.cluster_bootstrap(values, clusters, 500, seed=0)
    assert set(np.round(boot, 6)) <= {0.0, 5.0, 10.0}


def test_bootstrap_deterministic_given_seed():
    rng = np.random.default_rng(0)
    v, c = rng.normal(size=60), np.repeat(np.arange(6), 10)
    assert np.array_equal(S.cluster_bootstrap(v, c, 200, 3), S.cluster_bootstrap(v, c, 200, 3))


def test_top_k_stable_order():
    assert S.top_k([1.0, 3.0, 2.0, 3.0], 2) == [1, 3]
