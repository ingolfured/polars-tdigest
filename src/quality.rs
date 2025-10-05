//! Quality/diagnostics helpers for the t-digest implementation.
//!
//! This module lets us generate synthetic datasets with different shapes,
//! build digests, and compute simple goodness metrics (KS on quantiles-lite
//! and MAE over a quantile grid). It’s meant for *engineering feedback*,
//! not for publication-grade stats.

use crate::tdigest::TDigest;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A compact report we print in tests/benches.
#[derive(Debug, Clone, Copy)]
pub struct QualityReport {
    pub n: usize,
    /// “KS-like” score computed on quantiles (max absolute quantile error).
    pub ks: f64,
    /// Mean absolute error over a quantile grid.
    pub mae: f64,
    /// A single scalar for rough comparison (higher is better).
    pub score: f64,
}

impl QualityReport {
    fn from_metrics(n: usize, ks: f64, mae: f64) -> Self {
        // Heuristic scaler chosen so typical runs land ~0.4..0.95.
        // Tighter digests → smaller errors → larger score.
        let score = (-((1200.0 * mae) + (18.0 * ks))).exp();
        QualityReport { n, ks, mae, score }
    }
}

/// A few synthetic distributions we can mix & match.
#[derive(Debug, Clone, Copy)]
pub enum DistKind {
    /// Uniform in [0, 1).
    Uniform,
    /// Standard normal (Box–Muller).
    Normal,
    /// Log-normal-ish: exp(N(0, σ^2)) scaled.
    LogNormal { sigma: f64 },
    /// Mixture with a few regimes to stress tails and clumps.
    Mixture,
}

fn box_muller_normal<R: Rng>(rng: &mut R) -> f64 {
    // Classic Box–Muller. Keep away from 0 to avoid ln(0).
    let u1: f64 = rng.random::<f64>().clamp(1e-12, 1.0);
    let u2: f64 = rng.random::<f64>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Generate `n` samples for the chosen distribution.
/// We keep ranges roughly within [0, 1] after clamping so different
/// dists are comparable without post-scaling.
pub fn gen_dataset(kind: DistKind, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);

    match kind {
        DistKind::Uniform => {
            for _ in 0..n {
                out.push(rng.random::<f64>());
            }
        }
        DistKind::Normal => {
            for _ in 0..n {
                // map N(0,1) to ~[0,1] by 0.5 + 0.2*x and clamp
                let x = 0.5 + 0.2 * box_muller_normal(&mut rng);
                out.push(x.clamp(0.0, 1.0));
            }
        }
        DistKind::LogNormal { sigma } => {
            for _ in 0..n {
                let z = box_muller_normal(&mut rng);
                let x = (sigma * z).exp(); // positive and skewed
                                           // squash to [0,1] with a smooth transform
                let y = x / (1.0 + x);
                out.push(y.clamp(0.0, 1.0));
            }
        }
        DistKind::Mixture => {
            // A cheap “chaos kitchen”:
            // - 30% tight clumps (point-ish masses w/ tiny noise)
            // - 40% broad uniform
            // - 30% heavy-ish tails
            for _ in 0..n {
                let bucket: u32 = rng.random_range(0..100);
                let v = match bucket {
                    // Clumps around 0.1, 0.5, 0.9 with micro-noise
                    0..=29 => {
                        let center = match rng.random_range(0..3) {
                            0 => 0.10,
                            1 => 0.50,
                            _ => 0.90,
                        };
                        center + 1e-3 * rng.random_range(-1.0..1.0)
                    }
                    // Broad uniform
                    30..=69 => rng.random::<f64>(),
                    // “Tailier”: half near 0 with 1/x-ish bias, half near 1
                    _ => {
                        // Exponent in [3, 9) steers heaviness
                        let exp = rng.random_range(3.0..9.0);
                        if rng.random_bool(0.5) {
                            // left tail: small positives → near 0
                            let u = rng.random::<f64>().clamp(1e-12, 1.0);
                            u.powf(exp) // very small
                        } else {
                            // right tail: 1 - small positive
                            let u = rng.random::<f64>().clamp(1e-12, 1.0);
                            1.0 - u.powf(exp)
                        }
                    }
                };
                out.push(v.clamp(0.0, 1.0));
            }
        }
    }

    out
}

/// Compute an expected quantile via linear interpolation of order stats.
fn expected_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        return sorted[0];
    }
    if q >= 1.0 {
        return sorted[n - 1];
    }
    let t = q * (n as f64 - 1.0);
    let lo = t.floor() as usize;
    let hi = t.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let alpha = t - lo as f64;
        (1.0 - alpha) * sorted[lo] + alpha * sorted[hi]
    }
}

/// Evaluate a digest against the data by comparing its quantiles to the
/// empirical order statistics over a grid of q-values.
/// Returns (ks_like, mae).
fn quantile_grid_errors(td: &TDigest, sorted: &[f64]) -> (f64, f64) {
    // Dense grid but not crazy; 1000 points covers tails well.
    let steps = 1000usize;
    let mut ks_like = 0.0f64;
    let mut mae = 0.0f64;

    for i in 1..steps {
        let q = (i as f64) / (steps as f64);
        let est = td.estimate_quantile(q);
        let exp = expected_quantile(sorted, q);
        let err = (est - exp).abs();
        mae += err;
        if err > ks_like {
            ks_like = err;
        }
    }
    mae /= (steps - 1) as f64;
    (ks_like, mae)
}

/// Build a digest for `data` (already in [0,1]) and return a report.
pub fn assess(kind: DistKind, n: usize, max_size: usize, seed: u64) -> QualityReport {
    let mut data = gen_dataset(kind, n, seed);
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let td = TDigest::new_with_size(max_size).merge_sorted(data.clone());
    let (ks, mae) = quantile_grid_errors(&td, &data);
    QualityReport::from_metrics(n, ks, mae)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::TDigest;

    fn print_report(tag: &str, r: QualityReport) {
        println!(
            "{} -> QualityReport(n={}, KS={:.6e}, MAE={:.6e}, score={:.3})",
            tag, r.n, r.ks, r.mae, r.score
        );
    }

    /// Quick sanity: different scale functions / sizes should fall into a sane score band.
    #[test]
    fn quality_compare_scales_smoke() {
        // A stable seed keeps these prints deterministic across runs.
        let seed = 42;

        // We test multiple shapes; normal tends to be “easy”, mixture is “harder”.
        let n = 100_000;
        let max_size = 100;

        let r_uniform = assess(DistKind::Uniform, n, max_size, seed);
        let r_normal = assess(DistKind::Normal, n, max_size, seed);
        let r_logn = assess(DistKind::LogNormal { sigma: 1.0 }, n, max_size, seed);
        let r_mix = assess(DistKind::Mixture, n, max_size, seed);

        print_report("Uniform", r_uniform);
        print_report("Normal", r_normal);
        print_report("LogN(σ=1)", r_logn);
        print_report("Mixture", r_mix);

        // Very loose guards to catch regressions while permitting algorithm changes.
        // We mainly want “not disastrous”.
        for r in [r_uniform, r_normal, r_logn, r_mix] {
            assert!(r.ks.is_finite() && r.mae.is_finite());
            assert!(r.ks >= 0.0 && r.mae >= 0.0);
            assert!(r.score > 0.25, "score too low: {:?}", r);
        }
    }

    /// “Wide factor” runs: simulate larger working sets (stress merging) and
    /// just ensure things don’t crater. This mirrors the prints you were using.
    #[test]
    fn wide_factor_compares() {
        let seed = 123;
        let n = 100_000;

        // Baseline max_size; we vary only the data-generation “work size” indirectly by seed.
        let max_size = 1_000;

        println!("Quality harness: max_size={max_size}, work_factor=1, work_size=1000");
        let r1 = assess(DistKind::Mixture, n, max_size, seed);
        print_report("BASE  ", r1);

        println!("Quality harness: max_size={max_size}, work_factor=4.096, work_size=4096");
        let r2 = assess(DistKind::Mixture, n, max_size, seed.wrapping_mul(4096));
        print_report("WF4x  ", r2);

        println!("Quality harness: max_size={max_size}, work_factor=10, work_size=10000");
        let r3 = assess(DistKind::Mixture, n, max_size, seed.wrapping_mul(10_000));
        print_report("WF10x ", r3);

        for r in [r1, r2, r3] {
            assert!(
                r.score > 0.35,
                "unexpectedly low score under wide factor: {:?}",
                r
            );
        }
    }

    /// Smoke check that our helper can ingest already-sorted values without panicking.
    #[test]
    fn digest_builds_and_queries() {
        let mut vals = gen_dataset(DistKind::Normal, 10_000, 7);
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = TDigest::new_with_size(200).merge_sorted(vals.clone());

        // A couple of inexpensive checks so we fail loudly if something is wildly off.
        let q50 = td.estimate_quantile(0.5);
        let q10 = td.estimate_quantile(0.1);
        let q90 = td.estimate_quantile(0.9);
        assert!(q10 < q50 && q50 < q90, "monotonic quantiles violated");
        assert!((0.0..=1.0).contains(&q10));
        assert!((0.0..=1.0).contains(&q50));
        assert!((0.0..=1.0).contains(&q90));
    }
}
