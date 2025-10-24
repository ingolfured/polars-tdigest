pub mod cdf;
pub mod centroids;
pub mod codecs;
pub mod quantile;
pub mod test_helpers;

// Internal building blocks
mod compressor;
mod merges;
mod scale;
mod singleton_policy;
#[allow(clippy::module_inception)]
mod tdigest;

// Public surface
pub use self::tdigest::{TDigest, TDigestBuilder, ScaleFamily, SingletonPolicy, Centroid};


// Opt-in tracing (cheap unless env var set)
#[macro_export]
macro_rules! ttrace {
    ($($arg:tt)*) => {
        if std::env::var("TDIGEST_TRACE").is_ok() {
            eprintln!($($arg)*);
        }
    }
}
