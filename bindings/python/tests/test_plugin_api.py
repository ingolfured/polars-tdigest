import polars as pl
import tdigest_rs as ptd


def test_plugin_quantile_small():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    df = df.select(ptd.tdigest("x", 100))
    q = df.select(ptd.quantile("x", 0.5)).item()
    assert q == 2.5


def test_plugin_cdf_vectorized():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    df = df.select(ptd.tdigest("x", 100))
    out = df.select(ptd.cdf("x", [0.0, 1.5, 3.0]).alias("cdf"))
    output_l = out["cdf"].to_list()
    assert output_l == [0.125, 0.5, 0.875]
