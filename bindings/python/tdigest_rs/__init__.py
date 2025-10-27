# __init__.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

# --- Compiled extension -------------------------------------------------------

try:
    # The compiled pyo3 module is named "tdigest_rs" (no underscore)
    from .tdigest_rs import TDigest, __version__  # type: ignore[attr-defined]
except Exception as e:
    raise ImportError(
        "Failed to import the compiled extension 'tdigest_rs'. " "Build it with: `uv run maturin develop -r -F python`."
    ) from e

# --- Polars plugin helpers (single-file, no submodule) ------------------------

if TYPE_CHECKING:  # for type checkers only; avoids hard dep at import time
    import polars as pl
    from polars.type_aliases import IntoExpr

# Directory containing the compiled plugin .so/.pyd (sits next to this file)
_PLUGIN_DIR = Path(__file__).resolve().parent


def _register():
    # Lazy import so top-level package doesn't hard-depend on Polars
    import polars as pl
    from polars.plugins import register_plugin_function

    return pl, register_plugin_function


def tdigest(
    expr: "IntoExpr",
    max_size: int = 100,
    scale: str = "k2",
    storage: str = "f64",
) -> "pl.Expr":
    """
    Build a TDigest column from `expr` using the registered plugin.

    Parameters
    ----------
    expr : IntoExpr
        Numeric column/expression.
    max_size : int, default 100
        Target digest capacity.
    scale : {"k1","k2","k3","quad"}, default "k2"
        Scale family used by the compressor.
    storage : {"f64","f32"}, default "f64"
        Centroid precision. f32 is smaller, f64 is more precise.
    """
    pl, register = _register()
    fn = {"f64": "tdigest", "f32": "_tdigest_f32"}[storage.strip().lower()]
    return register(
        plugin_path=_PLUGIN_DIR,
        function_name=fn,
        args=[expr],
        kwargs={"max_size": max_size, "scale": scale},
        is_elementwise=False,
        returns_scalar=False,
    )


def quantile(expr: "IntoExpr", q: float) -> "pl.Expr":
    """
    Evaluate the q-quantile of a TDigest (scalar per group/partition).
    """
    pl, register = _register()
    return register(
        plugin_path=_PLUGIN_DIR,
        function_name="quantile",
        args=[expr, q],
        kwargs={},
        is_elementwise=False,
        returns_scalar=True,
    )


def cdf(
    expr: "IntoExpr",
    values: Union[float, int, Sequence[float], "pl.Series", "pl.Expr"],
) -> "pl.Expr":
    """
    Evaluate CDF(x) for one or many x; returns a list/series result.
    """
    pl, register = _register()
    return register(
        plugin_path=_PLUGIN_DIR,
        function_name="cdf",
        args=[expr, values],
        kwargs={},
        is_elementwise=False,
        returns_scalar=False,
    )


def median(expr: "IntoExpr") -> "pl.Expr":
    """
    Convenience: quantile at q=0.5 (scalar).
    """
    pl, register = _register()
    return register(
        plugin_path=_PLUGIN_DIR,
        function_name="median",
        args=[expr],
        kwargs={},
        is_elementwise=False,
        returns_scalar=True,
    )


def merge_tdigests(expr: "IntoExpr") -> "pl.Expr":
    """
    Merge a column of TDigest objects into a single TDigest (per group/partition).
    """
    pl, register = _register()
    return register(
        plugin_path=_PLUGIN_DIR,
        function_name="merge_tdigests",
        args=[expr],
        kwargs={},
        is_elementwise=False,
        returns_scalar=False,  # returns a TDigest object/series, not a scalar
    )


__all__ = [
    "TDigest",
    "__version__",
    # Polars helpers at top-level (no submodule)
    "tdigest",
    "quantile",
    "cdf",
    "median",
    "merge_tdigests",
]
