use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::tdigest::{ScaleFamily, TDigest as CoreTDigest, SingletonPolicy};

use bincode::config;
use bincode::serde::{decode_from_slice, encode_to_vec};

fn parse_scale(s: Option<&str>) -> Result<ScaleFamily, PyErr> {
    match s.map(|t| t.to_ascii_lowercase()) {
        None => Ok(ScaleFamily::K2), // default: K2 to match library-wide default
        Some(ref v) if v == "quad" => Ok(ScaleFamily::Quad),
        Some(ref v) if v == "k1" => Ok(ScaleFamily::K1),
        Some(ref v) if v == "k2" => Ok(ScaleFamily::K2),
        Some(ref v) if v == "k3" => Ok(ScaleFamily::K3),
        Some(v) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid scale: {v} (expected 'quad', 'k1', 'k2', or 'k3')"
        ))),
    }
}

#[pyclass(name = "TDigest")]
pub struct PyTDigest {
    inner: CoreTDigest,
}

#[pymethods]
impl PyTDigest {
    /// Build from a Python array-like of floats.
    /// Example: TDigest.from_array(xs, max_size=200, scale="k2")
    #[staticmethod]
    #[pyo3(signature = (xs, max_size=1000, scale=None))]
    pub fn from_array(xs: Vec<f64>, max_size: usize, scale: Option<&str>) -> PyResult<Self> {
        if max_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_size must be > 0",
            ));
        }
        let sc = parse_scale(scale)?;

        // Build an empty digest with the chosen scale, then merge the data.
        // SingletonPolicy::Off keeps Python side simple/predictable.
        let base = CoreTDigest::new(
            Vec::new(),         // centroids
            0.0,                // sum
            0.0,                // count
            sc,                 // scale
            SingletonPolicy::Off,
            max_size,
        );

        Ok(Self { inner: base.merge_unsorted(xs) })
    }

    pub fn median(&self) -> PyResult<f64> {
        Ok(self.inner.estimate_quantile(0.5))
    }

    pub fn quantile(&self, q: f64) -> PyResult<f64> {
        Ok(self.inner.estimate_quantile(q))
    }

    pub fn cdf(&self, py: Python<'_>, x: PyObject) -> PyResult<PyObject> {
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (x,))?;
        let xs: Vec<f64> = arr.extract()?;
        let ys: Vec<f64> = self.inner.estimate_cdf(&xs);
        let out = np.call_method1("asarray", (ys,))?;
        Ok(out.unbind())
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let cfg = config::standard();
        let bytes = encode_to_vec(&self.inner, cfg).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("serialize error: {e}"))
        })?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_bytes(b: Bound<'_, PyBytes>) -> PyResult<Self> {
        let cfg = config::standard();
        let (inner, _len): (CoreTDigest, usize) =
            decode_from_slice(b.as_bytes(), cfg).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("deserialize error: {e}"))
            })?;
        Ok(Self { inner })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTDigest>()?;
    Ok(())
}
