import polars as pl

from tdigest_rs import (
    tdigest,
    quantile,
    ScaleFamily,
    StorageSchema,
)


def _centroid_means_dtype(df: pl.DataFrame, storage) -> pl.DataType:
    out = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), max_size=64, scale=ScaleFamily.QUAD, storage=storage).alias("td"))
        .collect()
    )
    means_dtype = (
        out.lazy()
        .select(pl.col("td").struct.field("centroids").alias("centroids"))
        .explode("centroids")
        .select(pl.col("centroids").struct.field("mean").alias("mean"))
        .collect()["mean"]
        .dtype
    )
    return means_dtype


def test_tdigest_storage_f32_inner_means_are_float32():
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 3.0]})
    dtype = _centroid_means_dtype(df, StorageSchema.F32)
    assert dtype == pl.Float32


def test_tdigest_storage_f64_inner_means_are_float64():
    df = pl.DataFrame({"g": ["a"] * 3, "x": [1.0, 2.0, 3.0]})
    dtype = _centroid_means_dtype(df, StorageSchema.F64)
    assert dtype == pl.Float64


def test_tdigest_f32_has_lower_precision_than_f64():
    df = pl.DataFrame({"g": ["a"] * 2000, "x": list(range(1, 2001))})
    max_size = 512  # larger k → tighter median with QUAD
    q_target = 0.5
    true_median = 1000.5

    q64 = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), storage=StorageSchema.F64, scale=ScaleFamily.QUAD, max_size=max_size).alias("td"))
        .select(quantile("td", q=q_target))
        .collect()
        .item()
    )

    q32 = (
        df.lazy()
        .group_by("g")
        .agg(tdigest(pl.col("x"), storage=StorageSchema.F32, scale=ScaleFamily.QUAD, max_size=max_size).alias("td"))
        .select(quantile("td", q=q_target))
        .collect()
        .item()
    )

    # close to the true median (now that k is big enough)
    assert abs(q64 - true_median) < 3.0
    assert abs(q32 - true_median) < 3.0

    # F32 should be same or a bit worse than F64 (lower precision)
    err64 = abs(q64 - true_median)
    err32 = abs(q32 - true_median)
    assert err32 >= err64, f"expected F32 ({err32}) ≥ F64 ({err64})"
    assert err32 / (err64 + 1e-12) < 10.0
