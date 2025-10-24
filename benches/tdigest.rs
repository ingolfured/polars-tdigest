use ordered_float::OrderedFloat;

use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::compressor::compress_into;
use crate::tdigest::merges::{KWayCentroidMerge, MergeByMean};
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;

use serde::{Deserialize, Serialize};

/// TDigest orchestration + public API.
///
/// # Design: “single pipeline, many producers”
///
/// All ingestion paths (raw values, existing digests) are funneled into the **same**
/// compression pipeline in `compressor::compress_into`. The pipeline has a *fixed*,
/// easy-to-reason-about sequence of stages:
///
/// 1) **Normalize**  
///    Source iterators (e.g., `MergeByMean`, `KWayCentroidMerge`) produce a stream of centroids
///    that is non-decreasing by mean and *may* include adjacent equal means.  
///    `normalize_stream` (called inside `compress_into`) enforces:
///      - non-decreasing means (panics if violated),
///      - **coalescing of adjacent equal means** into a *singleton pile* (weight>1, `singleton=true`),
///      - running totals (∑w, ∑w·mean) and (min,max) for TDigest metadata.
///
///    Normalization is the **sole owner** of “equal means ⇒ pile” semantics—producers
///    never fuse and never guess the singleton bit.
///
/// 2) **Slice (edges vs. interior)**  
///    A `SingletonPolicy` is mapped to a lightweight `Policy` enum which computes slices:
///      - `left` (possibly protected edge items),
///      - `interior` (to be k-limited / budgeted),
///      - `right` (possibly protected edge items),
///      together with capacity rules (`EdgeCaps`) describing whether cap applies to the *core only*
///      or the *whole* digest.
///
/// 3) **Merge (k-limit)**  
///    The interior is greedily merged under the scale family rule `Δk ≤ 1`. Single-item
///    clusters that start with a singleton remain singletons; multi-item clusters are mixed.
///    This preserves order and total weight while concentrating mass near the center.
///
/// 4) **Cap (equal-weight bucketization)**  
///    If the interior overflows its budget, it is bucketized into approximately equal-weight
///    groups (order-preserving). **We intentionally allow emitting fewer than N buckets** when
///    the last partial bucket is small—this is numerically *better* in practice: fewer very small
///    centroids means less quantile jitter and better stability in tails. If an exact-N split is
///    ever required, a split-aware variant can be added, but the default favors precision.
///
/// 5) **Assemble**  
///    Edge slices are concatenated with the (capped) interior without mutation.
///
/// 6) **Post (policy-specific finalization)**  
///    Most policies are done; `Use` enforces a *total* cap (edges included) by applying the
///    same equal-weight bucketizer to the assembled result if necessary.
///
/// The result is then written back into the `TDigest`. Throughout, we preserve:
///   - strict ordering of centroid means (no duplicates),
///   - total weight (up to expected floating rounding),
///   - clear, local ownership of semantics (coalescing lives in Normalize; tail protection in Slice).
///
/// This structure keeps behavior deterministic and testable while making it obvious **where**
/// to change things (e.g., scale families only affect stage 3; tail policies only affect stage 2/6).
///
/// # Public API notes
/// - `merge_sorted`/`merge_unsorted` ingest raw values using `MergeByMean`.
/// - `merge_digests` ingests multiple digests using `KWayCentroidMerge`.
/// - All paths end up in the same compression pipeline.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct TDigest {
    centroids: Vec<Centroid>,
    max_size: usize,
    sum: OrderedFloat<f64>,
    count: OrderedFloat<f64>,
    max: OrderedFloat<f64>,
    min: OrderedFloat<f64>,
    scale: ScaleFamily,
    policy: SingletonPolicy,
}

impl Default for TDigest {
    fn default() -> Self {
        TDigest {
            centroids: Vec::new(),
            max_size: 1000, // default compression parameter
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: ScaleFamily::K2,       // default scale
            policy: SingletonPolicy::Use, // default singleton policy
        }
    }
}

/// Fluent builder.
#[derive(Debug, Clone)]
pub struct TDigestBuilder {
    max_size: usize,
    scale: ScaleFamily,
    policy: SingletonPolicy,
}

impl Default for TDigestBuilder {
    fn default() -> Self {
        TDigestBuilder {
            max_size: 1000,
            scale: ScaleFamily::K2,
            policy: SingletonPolicy::Use,
        }
    }
}

impl TDigestBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn max_size(mut self, n: usize) -> Self {
        self.max_size = n;
        self
    }
    pub fn scale(mut self, s: ScaleFamily) -> Self {
        self.scale = s;
        self
    }
    pub fn singleton_policy(mut self, p: SingletonPolicy) -> Self {
        self.policy = p;
        self
    }
    pub fn build(self) -> TDigest {
        TDigest {
            centroids: Vec::new(),
            max_size: self.max_size,
            sum: 0.0.into(),
            count: 0.0.into(),
            max: f64::NAN.into(),
            min: f64::NAN.into(),
            scale: self.scale,
            policy: self.policy,
        }
    }
}

impl TDigest {
    /// Entry point for fluent construction.
    pub fn builder() -> TDigestBuilder {
        TDigestBuilder::default()
    }

    /// Legacy conveniences (kept for low churn).
    pub fn new_with_size(max_size: usize) -> Self {
        Self::builder().max_size(max_size).build()
    }
    pub fn new_with_size_and_scale(max_size: usize, scale: ScaleFamily) -> Self {
        Self::builder().max_size(max_size).scale(scale).build()
    }

    #[inline] pub(crate) fn set_sum(&mut self, s: f64)   { self.sum = OrderedFloat(s); }
    #[inline] pub(crate) fn set_count(&mut self, c: f64) { self.count = OrderedFloat(c); }
    #[inline] pub(crate) fn set_min(&mut self, v: f64)   { self.min = OrderedFloat(v); }
    #[inline] pub(crate) fn set_max(&mut self, v: f64)   { self.max = OrderedFloat(v); }

    /// Build from unsorted values (convenience over `merge_unsorted`).
    pub fn from_unsorted(values: &[f64], max_size: usize) -> TDigest {
        let base = TDigest::builder().max_size(max_size).build();
        base.merge_unsorted(values.to_vec())
    }

    /// Convenience: estimate the q-quantile via method in `quantile.rs`.
    #[inline]
    pub fn quantile(&self, q: f64) -> f64 { self.estimate_quantile(q) }

    /// Convenience: estimate CDF(x) via method in `cdf.rs`.
    #[inline]
    pub fn cdf(&self, x: &[f64]) -> Vec<f64> { self.estimate_cdf(x) }

    /// Convenience: median == quantile(0.5).
    #[inline]
    pub fn median(&self) -> f64 { self.estimate_quantile(0.5) }

    #[inline] pub fn scale(&self) -> ScaleFamily { self.scale }
    #[inline] pub fn singleton_policy(&self) -> SingletonPolicy { self.policy }

    /// Ingest unsorted values; stable behavior identical to `merge_sorted` after sorting.
    pub fn merge_unsorted(&self, mut unsorted_values: Vec<f64>) -> TDigest {
        unsorted_values.sort_by(|a, b| a.total_cmp(b));
        self.merge_sorted(unsorted_values)
    }

    /// Ingest sorted values by interleaving with the existing centroids and running the pipeline.
    pub fn merge_sorted(&self, sorted_values: Vec<f64>) -> TDigest {
        if sorted_values.is_empty() {
            return self.clone();
        }
        let mut result = self.new_result_for_values(&sorted_values);

        // Producer: ordered stream of centroids + value runs (no coalescing here).
        let stream = MergeByMean::from_centroids_and_values(
            &self.centroids,
            &sorted_values,
            // mark value runs as singletons unless policy says Off
            !matches!(self.policy, SingletonPolicy::Off),
        );

        // Pipeline: Normalize → Slice → Merge(k-limit) → Cap → Assemble → Post
        let compressed = compress_into(&mut result, self.max_size, stream);
        result.centroids = compressed;

        // Strong invariant: strictly increasing means (no duplicates).
        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge"
        );
        result
    }

    /// Merge multiple digests by k-way merging their centroid runs and sending them through
    /// the exact same pipeline as raw values.
    pub fn merge_digests(digests: Vec<TDigest>) -> TDigest {
        // Decide max_size/scale/policy by first non-empty digest to keep semantics stable.
        let mut chosen = TDigest::default();
        let mut runs: Vec<&[Centroid]> = Vec::with_capacity(digests.len());
        let mut total_count = 0.0;
        let mut min = OrderedFloat::from(f64::INFINITY);
        let mut max = OrderedFloat::from(f64::NEG_INFINITY);

        for d in &digests {
            let n = d.count();
            if n > 0.0 && !d.centroids.is_empty() {
                if runs.is_empty() {
                    chosen.max_size = d.max_size;
                    chosen.scale = d.scale;
                    chosen.policy = d.policy;
                }
                total_count += n;
                min = std::cmp::min(min, d.min);
                max = std::cmp::max(max, d.max);
                runs.push(&d.centroids);
            }
        }
        if total_count == 0.0 {
            return TDigest::default();
        }

        let mut result = TDigest {
            centroids: Vec::new(),
            max_size: chosen.max_size,
            sum: 0.0.into(),
            count: total_count.into(),
            max,
            min,
            scale: chosen.scale,
            policy: chosen.policy,
        };

        // Producer: k-way merge of centroid runs (no coalescing).
        let merged_stream = KWayCentroidMerge::new(runs);

        // Same pipeline as values.
        let compressed = compress_into(&mut result, chosen.max_size, merged_stream);
        result.centroids = compressed;

        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge_digests"
        );
        result
    }

    /* ===========================
     * Small utilities
     * =========================== */

    pub fn new(
        centroids: Vec<Centroid>,
        sum: f64,
        count: f64,
        max: f64,
        min: f64,
        max_size: usize,
    ) -> Self {
        if centroids.len() <= max_size {
            TDigest {
                centroids,
                max_size,
                sum: OrderedFloat::from(sum),
                count: OrderedFloat::from(count),
                max: OrderedFloat::from(max),
                min: OrderedFloat::from(min),
                scale: ScaleFamily::K2,
                policy: SingletonPolicy::Use,
            }
        } else {
            let sz = centroids.len();
            let digests = vec![
                TDigest::new_with_size(100),
                TDigest::new(centroids, sum, count, max, min, sz),
            ];
            Self::merge_digests(digests)
        }
    }

    #[inline] pub fn mean(&self) -> f64 {
        let n = self.count.into_inner();
        if n > 0.0 { self.sum.into_inner() / n } else { 0.0 }
    }
    #[inline] pub fn sum(&self) -> f64   { self.sum.into_inner() }
    #[inline] pub fn count(&self) -> f64 { self.count.into_inner() }
    #[inline] pub fn max(&self) -> f64   { self.max.into_inner() }
    #[inline] pub fn min(&self) -> f64   { self.min.into_inner() }
    #[inline] pub fn is_empty(&self) -> bool { self.centroids.is_empty() }
    #[inline] pub fn max_size(&self) -> usize { self.max_size }
    #[inline] pub fn centroids(&self) -> &[Centroid] { &self.centroids }

    fn new_result_for_values(&self, values: &[f64]) -> TDigest {
        let mut r = TDigest {
            centroids: Vec::new(),
            max_size: self.max_size(),
            sum: 0.0.into(),
            count: (self.count() + values.len() as f64).into(),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: self.scale,
            policy: self.policy,
        };

        let vmin = OrderedFloat::from(values[0]);
        let vmax = OrderedFloat::from(values[values.len() - 1]);
        if self.count() > 0.0 {
            r.min = std::cmp::min(self.min, vmin);
            r.max = std::cmp::max(self.max, vmax);
        } else {
            r.min = vmin;
            r.max = vmax;
        }
        r
    }
}
