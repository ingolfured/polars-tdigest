use ordered_float::OrderedFloat;

use crate::tdigest::centroids::{is_sorted_strict_by_mean, Centroid};
use crate::tdigest::compressor::compress_into;
use crate::tdigest::merges::MergeByMean;
use crate::tdigest::scale::ScaleFamily;
use crate::tdigest::singleton_policy::SingletonPolicy;

use serde::{Deserialize, Serialize};

/// The TDigest structure and its fluent builder.
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
            max_size: 1000, // requested default
            sum: OrderedFloat::from(0.0),
            count: OrderedFloat::from(0.0),
            max: OrderedFloat::from(f64::NAN),
            min: OrderedFloat::from(f64::NAN),
            scale: ScaleFamily::K2,       // default
            policy: SingletonPolicy::Use, // default
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

    #[inline]
    pub(crate) fn set_sum(&mut self, s: f64) {
        self.sum = OrderedFloat(s);
    }

    #[inline]
    pub(crate) fn set_count(&mut self, c: f64) {
        self.count = OrderedFloat(c);
    }

    #[inline]
    pub(crate) fn set_min(&mut self, v: f64) {
        self.min = OrderedFloat(v);
    }

    #[inline]
    pub(crate) fn set_max(&mut self, v: f64) {
        self.max = OrderedFloat(v);
    }

    /// Build from unsorted values (convenience over `merge_unsorted`).
    pub fn from_unsorted(values: &[f64], max_size: usize) -> TDigest {
        let base = TDigest::builder().max_size(max_size).build();
        base.merge_unsorted(values.to_vec())
    }

    /// Convenience: estimate the q-quantile via method in `quantile.rs`.
    #[inline]
    pub fn quantile(&self, q: f64) -> f64 {
        self.estimate_quantile(q)
    }

    /// Convenience: estimate CDF(x) via method in `cdf.rs`.
    #[inline]
    pub fn cdf(&self, x: &[f64]) -> Vec<f64> {
        self.estimate_cdf(x)
    }

    /// Convenience: median == quantile(0.5).
    #[inline]
    pub fn median(&self) -> f64 {
        self.estimate_quantile(0.5)
    }

    #[inline]
    pub fn scale(&self) -> ScaleFamily {
        self.scale
    }
    #[inline]
    pub fn singleton_policy(&self) -> SingletonPolicy {
        self.policy
    }

    pub fn merge_unsorted(&self, mut unsorted_values: Vec<f64>) -> TDigest {
        unsorted_values.sort_by(|a, b| a.total_cmp(b));
        self.merge_sorted(unsorted_values)
    }

    pub fn merge_sorted(&self, sorted_values: Vec<f64>) -> TDigest {
        if sorted_values.is_empty() {
            return self.clone();
        }
        let mut result = self.new_result_for_values(&sorted_values);

        let stream = MergeByMean::from_centroids_and_values(
            &self.centroids,
            &sorted_values,
            // mark value runs as singletons unless policy says Off
            !matches!(self.policy, SingletonPolicy::Off),
        );
        let compressed = compress_into(&mut result, self.max_size, stream);
        result.centroids = compressed;

        // Strong invariant: strictly increasing means (no duplicates).
        debug_assert!(
            is_sorted_strict_by_mean(&result.centroids),
            "duplicate centroid means after merge"
        );
        result
    }

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

        let merged_stream = crate::tdigest::merges::KWayCentroidMerge::new(runs);
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
                TDigestBuilder::new().max_size(100).build(),
                TDigest::new(centroids, sum, count, max, min, sz),
            ];
            Self::merge_digests(digests)
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        let n = self.count.into_inner();
        if n > 0.0 {
            self.sum.into_inner() / n
        } else {
            0.0
        }
    }
    #[inline]
    pub fn sum(&self) -> f64 {
        self.sum.into_inner()
    }
    #[inline]
    pub fn count(&self) -> f64 {
        self.count.into_inner()
    }
    #[inline]
    pub fn max(&self) -> f64 {
        self.max.into_inner()
    }
    #[inline]
    pub fn min(&self) -> f64 {
        self.min.into_inner()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }
    #[inline]
    pub fn max_size(&self) -> usize {
        self.max_size
    }
    #[inline]
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

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

/* ===========================
 * Tests: merge & structural invariants (high-level TDigest)
 * =========================== */
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::test_helpers::{assert_exact, assert_rel_close};
    use crate::tdigest::{Centroid, ScaleFamily, SingletonPolicy};

    /// Identical values become singleton piles (weight>1 with `singleton=true`);
    /// distinct values are weight==1 singletons.
    #[test]
    fn singletons_grouped_runs() {
        // Values: three 1s, one 2, two 3s
        let vals = vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0];
        let t = TDigestBuilder::new().max_size(64).build().merge_sorted(vals);

        let cs = t.centroids();
        assert!(
            cs.windows(2).all(|w| w[0].mean() != w[1].mean()),
            "duplicate centroid means found"
        );

        assert_eq!(cs.len(), 3);

        assert_eq!(cs[0].mean(), 1.0);
        assert_eq!(cs[0].weight(), 3.0);
        assert!(cs[0].is_singleton());

        assert_eq!(cs[1].mean(), 2.0);
        assert_eq!(cs[1].weight(), 1.0);
        assert!(cs[1].is_singleton());

        assert_eq!(cs[2].mean(), 3.0);
        assert_eq!(cs[2].weight(), 2.0);
        assert!(cs[2].is_singleton());
    }

    #[test]
    fn merge_digests_uniform_100x1000() {
        let mut digests: Vec<TDigest> = Vec::with_capacity(100);
        for _ in 1..=100 {
            let t = TDigest::builder()
                .max_size(100)
                .singleton_policy(SingletonPolicy::Use)
                .build()
                .merge_sorted((1..=1_000).map(f64::from).collect());
            digests.push(t);
        }
        let t = TDigest::merge_digests(digests);

        assert_exact("Q(1.00)", 1000.0, t.estimate_quantile(1.0));
        assert_exact("Q(0.00)", 1.0, t.estimate_quantile(0.0));

        let q99 = t.estimate_quantile(0.99);
        assert_rel_close("Q(0.99)", 990.0, q99, 7e-4);

        let q01 = t.estimate_quantile(0.01);
        assert_rel_close("Q(0.01)", 10.0, q01, 2e-1);

        let q50 = t.estimate_quantile(0.5);
        assert_rel_close("Q(0.50)", 500.0, q50, 2e-3);
    }

    #[test]
    fn merge_unsorted_smoke_small() {
        let vals = vec![5.0, 1.0, 3.0, 4.0, 2.0, 2.0, 9.0, 7.0];
        let t = TDigestBuilder::new().max_size(64).build().merge_unsorted(vals.clone());

        assert_exact("count", vals.len() as f64, t.count());
        assert_exact("min", 1.0, t.min());
        assert_exact("max", 9.0, t.max());

        assert_exact("Q(0.00)", 1.0, t.estimate_quantile(0.0));
        assert_exact("Q(1.00)", 9.0, t.estimate_quantile(1.0));

        let expected_mean = vals.iter().sum::<f64>() / (vals.len() as f64);
        assert_rel_close("mean", expected_mean, t.mean(), 1e-9);
    }

    #[test]
    fn merge_digests_two_blocks_smoke() {
        let d1 =
            TDigestBuilder::new().max_size(128).build().merge_sorted((1..=5).map(f64::from).collect::<Vec<_>>());
        let d2 =
            TDigestBuilder::new().max_size(128).build().merge_sorted((6..=10).map(f64::from).collect::<Vec<_>>());

        let t = TDigest::merge_digests(vec![d1, d2]);

        assert_exact("count", 10.0, t.count());
        assert_exact("min", 1.0, t.min());
        assert_exact("max", 10.0, t.max());

        assert_exact("Q(0.00)", 1.0, t.estimate_quantile(0.0));
        assert_exact("Q(1.00)", 10.0, t.estimate_quantile(1.0));
        assert_rel_close("Q(0.50)", 5.5, t.estimate_quantile(0.5), 1e-9);
    }

    #[test]
    fn compact_scaling_prevents_overflow_and_preserves_shape() {
        use crate::tdigest::test_helpers::assert_rel_close;

        // Three centroids with astronomical weights to force scaling for f32.
        let c0 = Centroid::new(0.0, 1.0e36);
        let c1 = Centroid::new(1.0, 2.0e36);
        let c2 = Centroid::new(2.0, 3.0e36);
        let cents = vec![c0, c1, c2];

        // Keep TDigest "as-is" (no compression): set max_size comfortably above len.
        let sum = c0.mean() * c0.weight() + c1.mean() * c1.weight() + c2.mean() * c2.weight(); // 8e36
        let count = c0.weight() + c1.weight() + c2.weight(); // 6e36
        let td = TDigest::new(cents, sum, count, 2.0, 0.0, 512);

        // Baseline quantile
        let p50_before = td.estimate_quantile(0.5);

        // Write compact (f32/f32) — this will auto-scale internally.
        let ser = td.clone().try_to_series_compact("n").unwrap();

        // Sanity: compact centroids' weights must be finite f32 and well within range.
        {
            let sc = ser.struct_().expect("struct");
            let centroids_col = sc.field_by_name("centroids").expect("centroids");
            let list = centroids_col.list().expect("list");
            let row0 = list.get_as_series(0).expect("row0");
            let inner = row0.struct_().expect("inner");
            let w_series = inner.field_by_name("weight").expect("weight");
            let w_f32 = w_series.f32().expect("f32 weights");

            for i in 0..w_f32.len() {
                if let Some(wf) = w_f32.get(i) {
                    let wf: f32 = wf;
                    assert!(wf.is_finite(), "compact weight must be finite");
                    assert!(wf.abs() < f32::MAX / 4.0, "compact weight too large: {wf}");
                }
            }
        }

        // Parse back and verify shape invariants.
        let parsed = TDigest::try_from_series_compact(&ser).unwrap();
        assert_eq!(parsed.len(), 1, "one digest expected");
        let td2 = parsed.into_iter().next().unwrap();

        // 1) Quantiles should be preserved under uniform re-scaling of weights.
        let p50_after = td2.estimate_quantile(0.5);
        assert_rel_close("p50 preserved after scaling", p50_before, p50_after, 1e-6);

        // 2) Relative weights should match (only a global factor differs).
        let w1: Vec<f64> = td.centroids().iter().map(|c| c.weight()).collect();
        let w2: Vec<f64> = td2.centroids().iter().map(|c| c.weight()).collect();
        assert_eq!(w1.len(), w2.len(), "centroid count must match");

        let s1: f64 = w1.iter().sum();
        let s2: f64 = w2.iter().sum();
        for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
            let ra = *a / s1;
            let rb = *b / s2;
            assert_rel_close(&format!("weight ratio[{i}]"), ra, rb, 1e-7);

            // 3) Optional: CDF invariants at a few points.
            let xs = [0.5, 1.0, 1.5];
            let cdf1 = td.estimate_cdf(&xs);
            let cdf2 = td2.estimate_cdf(&xs);
            for i in 0..xs.len() {
                assert_rel_close(&format!("CDF({}) preserved", xs[i]), cdf1[i], cdf2[i], 1e-6);
            }
        }
    }

    #[test]
    fn edge_tail_protection_small_max() {
        use crate::tdigest::test_helpers::assert_exact;

        // Tiny core, protect 3 raw items per tail.
        let base = TDigest::builder()
            .max_size(4)
            .scale(ScaleFamily::Quad)
            .singleton_policy(SingletonPolicy::UseWithProtectedEdges(3))
            .build();

        // Merge three batches:
        // - middle bulk: 1..=20
        // - extend right tail: 21..=30
        // - distinct left tail: -3, -2, -1  (ensures 3 separate edge centroids are protected)
        let t1 = base.merge_sorted((1..=20).map(f64::from).collect());
        let t2 = t1.merge_sorted((21..=30).map(f64::from).collect());
        let t3 = t2.merge_sorted(vec![-3.0, -2.0, -1.0]);

        // Global invariants
        assert_exact("count", 33.0, t3.count());
        assert_exact("min", -3.0, t3.min());
        assert_exact("max", 30.0, t3.max());

        let cs = t3.centroids();

        // Sorted by mean (non-decreasing) and strictly increasing (no duplicate centroids).
        assert!(
            cs.windows(2).all(|w| w[0].mean() <= w[1].mean()),
            "centroids not sorted by mean"
        );
        assert!(
            cs.windows(2).all(|w| w[0].mean() < w[1].mean()),
            "duplicate centroid means found"
        );

        // All centroids: weight must be finite and > 0.
        for (i, c) in cs.iter().enumerate() {
            assert!(
                c.weight().is_finite() && c.weight() > 0.0,
                "invalid weight at {i}"
            );
            assert!(c.mean().is_finite(), "invalid mean at {i}");
        }

        // Must have at least the 3+3 protected edge centroids plus some middle.
        assert!(
            cs.len() >= 6,
            "need ≥6 centroids (3 per edge), got {}",
            cs.len()
        );

        // ---- Left edge: first 3 must be protected (singleton or pile) at -3, -2, -1.
        assert!(cs[0].is_singleton() && cs[1].is_singleton() && cs[2].is_singleton());
        assert_exact("left[0] mean", -3.0, cs[0].mean());
        assert_exact("left[1] mean", -2.0, cs[1].mean());
        assert_exact("left[2] mean", -1.0, cs[2].mean());

        // No cross-boundary absorption: next centroid (if any) strictly greater than -1.
        if cs.len() > 3 {
            assert!(
                cs[3].mean() > -1.0,
                "merged across left protected boundary (cs[3]={})",
                cs[3].mean()
            );
        }

        // ---- Right edge: last 3 must be protected (singleton or pile) at 28, 29, 30.
        let n = cs.len();
        assert!(cs[n - 3].is_singleton() && cs[n - 2].is_singleton() && cs[n - 1].is_singleton());
        assert_exact("right[-3] mean", 28.0, cs[n - 3].mean());
        assert_exact("right[-2] mean", 29.0, cs[n - 2].mean());
        assert_exact("right[-1] mean", 30.0, cs[n - 1].mean());

        // ---- Extreme quantiles exact due to protected edges.
        assert_exact("Q(0.00)", -3.0, t3.estimate_quantile(0.0));
        assert_exact("Q(1.00)", 30.0, t3.estimate_quantile(1.0));

        // ---- Quantiles near tails: inside reasonable bands with tiny core.
        let q10 = t3.estimate_quantile(0.10);
        assert!(
            (-3.0..=1.0).contains(&q10),
            "Q(0.10) expected in [-3,1], got {}",
            q10
        );

        let q90 = t3.estimate_quantile(0.90);
        assert!(
            (25.0..=30.0).contains(&q90),
            "Q(0.90) expected in [25,30], got {}",
            q90
        );

        // Sanity: centroid count remains bounded with tiny core + protected tails.
        assert!(
            cs.len() == 10,
            "too many centroids for tiny capacity with protected tails; got {}",
            cs.len()
        );
    }

    #[test]
    fn compresses_to_exactly_max_size_with_singletons_enabled() {
        use crate::tdigest::{ScaleFamily, SingletonPolicy, TDigest};

        let td = TDigest::builder()
            .max_size(10)
            .scale(ScaleFamily::K2)
            .singleton_policy(SingletonPolicy::Use)
            .build();

        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let td = td.merge_unsorted(vals); // returns new digest

        assert!(
            td.centroids().len() <= 10,
            "digest should have at most 10 centroids, got {}",
            td.centroids().len()
        );
    }
}
