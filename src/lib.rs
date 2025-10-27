#![allow(non_snake_case)]

//! TDigest for Rust with optional Python and Java bindings.
//!
//! Features
//! - python: builds a CPython extension module called `tdigest_rs`.
//! - java: enables JNI bindings in the `jni` module.
//! - On Linux, the global allocator is set to jemalloc to reduce fragmentation
//!   and improve multi-threaded performance.
//!
//! Python (dev loop)
//!   uv run maturin develop -r -F python
//!   python -c "import tdigest_rs; print(tdigest_rs.__version__)"
//!
//! Jemalloc (Linux-only)
//! We set jemalloc as the global allocator on Linux targets. If you prefer the
//! system allocator, remove the `jemallocator` dependency and the
//! `#[global_allocator]` block below.

mod polars_expr;
pub mod tdigest;

#[cfg(feature = "java")]
pub mod jni;

// ---- test-only quality modules ----------------------------------------------
#[cfg(test)]
pub mod quality {
    pub mod cdf_quality;
    pub mod quality_base;
    pub mod quantile_quality;
}

#[cfg(test)]
pub use crate::quality::quality_base::{print_banner, print_report, print_section};

// ---- jemalloc on linux (kept intentionally) ---------------------------------
#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

#[cfg(feature = "python")]
mod py;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

/// CPython module entry point: `PyInit_tdigest_rs`.
///
/// The Rust function name determines the init symbol name, so keeping it
/// identical to the Python module name avoids surprises. We attach a
/// `__version__` from Cargo metadata and delegate registrations to `py::register`.
#[cfg(feature = "python")]
#[pymodule]
// #[pyo3(name = "_tdigest_rs")]
fn tdigest_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
