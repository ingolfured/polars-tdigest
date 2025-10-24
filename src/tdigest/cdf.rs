// src/tdigest/cdf.rs
use crate::tdigest::TDigest;
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

/// Crossover for parallel evaluation with Rayon.
/// Keep conservative to avoid overhead dominating small batches.
const PAR_MIN: usize = 32_768;

impl TDigest {
    /// Estimate the CDF at each `vals[i]`, returning values in [0, 1].
    ///
    /// Semantics closely follow the reference MergingDigest.cdf():
    /// - Outside support: clamp to {0, 1}.
    /// - Left/right tails: guarded linear ramps between min↔first-mean and last-mean↔max,
    ///   with half-weight treatment at the edge centroids.
    /// - Exact hits at centroid means: midpoint (prefix + 0.5*w)/N.
    /// - Between centroids: center-to-center interpolation with singleton exclusion
    ///   (unit centroids don't contribute half-weight to interpolation span).
    pub fn estimate_cdf(&self, vals: &[f64]) -> Vec<f64> {
        // Degenerate cases
        let n = self.centroids().len();
        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        if vals.is_empty() {
            return Vec::new();
        }

        // Build "light arrays" once per call.
        let CdfLight {
            means,
            weights,
            prefix,
            count_f64,
        } = build_arrays_light(self);

        let min_v = self.min();
        let max_v = self.max();

        // Tiny fast path: ≤ 8 queries, scalar loop (hot for point lookups).
        if vals.len() <= 8 {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                out.push(cdf_at_val_fast(
                    v, &means, &weights, &prefix, count_f64, min_v, max_v,
                ));
            }
            return out;
        }

        // Main path: parallel for big batches, scalar otherwise.
        if vals.len() >= PAR_MIN {
            vals.par_iter()
                .with_min_len(4096)
                .map(|&v| cdf_at_val_fast(v, &means, &weights, &prefix, count_f64, min_v, max_v))
                .collect()
        } else {
            let mut out = Vec::with_capacity(vals.len());
            for &v in vals {
                out.push(cdf_at_val_fast(
                    v, &means, &weights, &prefix, count_f64, min_v, max_v,
                ));
            }
            out
        }
    }
}

/* ------------------------- PRIVATE HELPERS ------------------------- */

struct CdfLight {
    means: Vec<f64>,
    weights: Vec<f64>,
    prefix: Vec<f64>,
    count_f64: f64,
}

#[inline]
fn build_arrays_light(td: &TDigest) -> CdfLight {
    let n = td.centroids().len();

    let mut means = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for c in td.centroids() {
        means.push(c.mean());
        weights.push(c.weight());
    }

    // prefix[i] = sum(weights[0..i])
    let mut prefix = Vec::with_capacity(n);
    let mut run = 0.0;
    for &w in &weights {
        prefix.push(run);
        run += w;
    }

    CdfLight {
        means,
        weights,
        prefix,
        count_f64: td.count(), // equals `run`, but use the accessor for clarity
    }
}

/* --------- Optimized evaluation kernel (reference semantics) ---------
Exact hit → (prefix[idx] + 0.5*w)/N.
Left/right tails → guarded ramps using min/max and edge centroid half-weights.
Between centroids → center-to-center interpolation with singleton exclusion. */
#[inline(always)]
fn cdf_at_val_fast(
    val: f64,
    means: &[f64],
    weights: &[f64],
    prefix: &[f64],
    count_f64: f64,
    min_v: f64,
    max_v: f64,
) -> f64 {
    let n = means.len();

    match means.binary_search_by(|m| m.partial_cmp(&val).unwrap()) {
        // Exact centroid hit: midpoint semantics (half-weight).
        Ok(idx) => (prefix[idx] + 0.5 * weights[idx]) / count_f64,

        Err(idx) => {
            // Left of first centroid mean
            if idx == 0 {
                if val < min_v {
                    return 0.0;
                }
                let m0 = means[0];
                let w0 = weights[0];
                let gap = m0 - min_v;
                if gap > 0.0 {
                    if val == min_v {
                        return 0.5 / count_f64;
                    }
                    // guarded ramp from min → mean[0] with half-weight at edge centroid
                    return (1.0 + (val - min_v) / gap * (w0 / 2.0 - 1.0)) / count_f64;
                } else {
                    // degenerate (all equal); but idx==0 implies val<=min, so:
                    return 0.0;
                }
            }

            // Right of last centroid mean
            if idx == n {
                if val > max_v {
                    return 1.0;
                }
                let mn = means[n - 1];
                let wn = weights[n - 1];
                let gap = max_v - mn;
                if gap > 0.0 {
                    if val == max_v {
                        return 1.0 - 0.5 / count_f64;
                    }
                    // guarded ramp from mean[last] → max with half-weight at edge centroid
                    let dq = (1.0 + (max_v - val) / gap * (wn / 2.0 - 1.0)) / count_f64;
                    return 1.0 - dq;
                } else {
                    return 1.0;
                }
            }

            // Between centroids idx-1 and idx
            let li = idx - 1;
            let ri = idx;
            let ml = means[li];
            let mr = means[ri];
            let wl = weights[li];
            let wr = weights[ri];

            let gap = mr - ml;
            if gap <= 0.0 {
                // Too close / pathological: fall back to midpoint mass
                let dw = 0.5 * (wl + wr);
                return (prefix[li] + dw) / count_f64;
            }

            // Two singletons bracketing → no interpolation: step over left singleton
            if wl == 1.0 && wr == 1.0 {
                return (prefix[li] + 1.0) / count_f64;
            }

            // Singleton exclusion (unit heaps are atomic; exclude 0.5 from interpolation span)
            let left_excl = if wl == 1.0 { 0.5 } else { 0.0 };
            let right_excl = if wr == 1.0 { 0.5 } else { 0.0 };

            let dw = 0.5 * (wl + wr);
            let dw_no_singleton = dw - left_excl - right_excl;
            // Safety (mirrors reference asserts)
            // assert dw_no_singleton > dw / 2.0 && gap > 0.0

            // Base mass at left half-weight + any left exclusion
            let base = prefix[li] + wl / 2.0 + left_excl;
            let frac = (val - ml) / gap;
            (base + dw_no_singleton * frac) / count_f64
        }
    }
}

/* --------- Reference exact ECDF for tests (midpoint semantics over ties) --------- */

#[allow(dead_code)]
fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    if n == 0 {
        return Vec::new();
    }

    let nf = n as f64;
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }
        let mid = (i + j) as f64 * 0.5;
        let val = mid / nf;
        out.extend(std::iter::repeat_n(val, j - i));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::tdigest::{tdigest::TDigestBuilder, ScaleFamily};

    #[test]
    fn cdf_small_max10_medn_100_simple() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect();
        values[0] = -1e9; // extreme left tail
        values[99] = 1e9; // extreme right tail
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let td =
            TDigestBuilder::new().max_size(10).scale(ScaleFamily::Quad).build().merge_sorted(values.clone());
        let approx = td.estimate_cdf(&values);
        assert_eq!(approx.len(), values.len());

        // 1) bounds + 2) monotone
        for i in 0..approx.len() {
            let p = approx[i];
            assert!((0.0..=1.0).contains(&p), "cdf[{}]={} out of [0,1]", i, p);
            if i > 0 {
                assert!(approx[i] + 1e-12 >= approx[i - 1], "non-monotone at {}", i);
            }
        }

        // 3) tails hit ~1%/~99% (allow tiny slack for compression)
        let eps = 1e-6;
        assert!(
            approx[0] <= 0.01 + eps,
            "left tail too large: {}",
            approx[0]
        );
        assert!(
            approx[99] >= 0.99 - eps,
            "right tail too small: {}",
            approx[99]
        );

        // 4) middle rank (~median) should be roughly around 0.5
        let mid = 50;
        assert!(
            (0.49..=0.51).contains(&approx[mid]),
            "median sanity: {}",
            approx[mid]
        );
    }

    #[test]
    fn cdf_exact_with_enough_capacity() {
        use crate::tdigest::test_helpers::assert_exact;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        const N: usize = 9_999;
        let mut rng = StdRng::seed_from_u64(42);
        let mut vals: Vec<f64> = (0..N)
            .map(|_| rng.random_range(0..N as u64) as f64)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let exact = super::exact_ecdf_for_sorted(&vals);
        let approx = TDigestBuilder::new().max_size(N + 1).build()
            .merge_sorted(vals.clone())
            .estimate_cdf(&vals);

        for (i, (&e, &a)) in exact.iter().zip(&approx).enumerate() {
            assert_exact(&format!("CDF[{i}]"), e, a);
        }
    }
}
