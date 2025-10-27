import math
import numpy as np
from tdigest_rs import TDigest


def test_class_basic_quantiles_and_median():
    t = TDigest.from_array([1.0, 2.0, 3.0, 4.0], max_size=32)
    assert t.median() == 2.5
    assert t.quantile(0.25) == 1.5


def test_class_cdf_vectorized():
    t = TDigest.from_array(np.arange(6, dtype=float), max_size=32)  # [0..5]
    values = [0.0, 2.5, 5.0]
    ys = t.cdf(values)
    assert len(ys) == 3
    assert ys[1] == 0.5


def test_class_cdf_vectorized_numpy():
    t = TDigest.from_array(np.arange(6, dtype=float), max_size=32)  # [0..5]
    values = np.array([0.0, 2.5, 5.0], dtype=float)
    ys = t.cdf(values)
    assert len(ys) == 3
    assert ys[1] == 0.5


def test_bytes_roundtrip():
    t = TDigest.from_array([10.0, 20.0, 30.0], max_size=32)
    b = t.to_bytes()
    t2 = TDigest.from_bytes(b)
    assert math.isfinite(t2.quantile(0.9))
    assert t.median() == t2.median() == 20.0
