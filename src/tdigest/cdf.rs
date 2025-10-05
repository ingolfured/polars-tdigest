// src/tdigest/cdf.rs
use super::TDigest;

/// Exact ECDF at each sorted sample (midpoint convention on ties).
/// Public so quality + tests can use it.
pub fn exact_ecdf_for_sorted(sorted: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    if n == 0 {
        return Vec::new();
    }

    let nf = n as f64;
    let mut out = Vec::with_capacity(n);

    let mut i = 0usize;
    while i < n {
        // advance to end of the run of equal values
        let mut j = i + 1;
        while j < n && sorted[j] == sorted[i] {
            j += 1;
        }

        // midpoint convention on ties
        let mid = (i + j) as f64 / 2.0;
        let val = mid / nf;
        out.extend(std::iter::repeat_n(val, j - i));

        i = j;
    }

    out
}

impl TDigest {
    // See
    // https://github.com/protivinsky/pytdigest/blob/main/pytdigest/tdigest.c#L300-L336
    pub fn estimate_cdf(&self, vals: &[f64]) -> Vec<f64> {
        let n = self.centroids.len();
        if n == 0 {
            return vec![f64::NAN; vals.len()];
        }
        let mut means: Vec<f64> = Vec::with_capacity(n);
        let mut weights: Vec<f64> = Vec::with_capacity(n);
        for c in &self.centroids {
            means.push(c.mean());
            weights.push(c.weight());
        }
        // Precompute running total weights
        let mut running_total_weights: Vec<f64> = Vec::with_capacity(n);
        let mut total = 0.0;
        for &w in &weights {
            running_total_weights.push(total);
            total += w;
        }
        let count = self.count();
        vals.iter()
            .map(|&val| {
                // Binary search for the first centroid with mean >= val
                match means.binary_search_by(|m| m.partial_cmp(&val).unwrap()) {
                    Ok(mut centroid_index) => {
                        // Find the first centroid with mean == val (in case of duplicates)
                        while centroid_index > 0 && means[centroid_index - 1] == val {
                            centroid_index -= 1;
                        }
                        // Sum all centroids with this mean
                        let mut weight_at_value = weights[centroid_index];
                        let running_total_start = running_total_weights[centroid_index];
                        let mut i = centroid_index + 1;
                        while i < n && means[i] == val {
                            weight_at_value += weights[i];
                            i += 1;
                        }
                        (running_total_start + (weight_at_value / 2.0)) / count
                    }
                    Err(centroid_index) => {
                        if centroid_index == 0 {
                            0.0
                        } else if centroid_index >= n {
                            1.0
                        } else {
                            let cr_mean = means[centroid_index];
                            let cl_mean = means[centroid_index - 1];
                            let cr_weight = weights[centroid_index];
                            let cl_weight = weights[centroid_index - 1];
                            let mut running_total_weight = running_total_weights[centroid_index];
                            running_total_weight -= cl_weight / 2.0;
                            // If both are weight 1 and val is exactly halfway, return mean of their CDF steps
                            if cl_weight == 1.0
                                && cr_weight == 1.0
                                && ((val - (cl_mean + cr_mean) / 2.0).abs() < 1e-12)
                            {
                                let cdf_left =
                                    (running_total_weights[centroid_index - 1] + 0.5) / count;
                                let cdf_right =
                                    (running_total_weights[centroid_index] + 0.5) / count;
                                return 0.5 * (cdf_left + cdf_right);
                            }
                            // Check if within 0.5 weighted distance of a centroid of weight 1
                            if cl_weight == 1.0 && (val - cl_mean).abs() <= 0.5 {
                                // Step at cl_mean
                                return (running_total_weights[centroid_index - 1] + 0.5) / count;
                            }
                            if cr_weight == 1.0 && (val - cr_mean).abs() <= 0.5 {
                                // Step at cr_mean
                                return (running_total_weights[centroid_index] + 0.5) / count;
                            }
                            let m = (cr_mean - cl_mean) / (cl_weight / 2.0 + cr_weight / 2.0);
                            let x = (val - cl_mean) / m;
                            (running_total_weight + x) / count
                        }
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdigest::TDigest;

    // Minimal helpers for CDF tests
    use crate::tdigest::test_helpers::*;
    fn assert_cdf_oob_clamps(label: &str, td: &crate::tdigest::TDigest, below: f64, above: f64) {
        let v = td.estimate_cdf(&[below, above]);
        assert!((v[0] - 0.0).abs() == 0.0, "{label}/min-ε: {}", v[0]);
        assert!((v[1] - 1.0).abs() == 0.0, "{label}/max+ε: {}", v[1]);
    }

    /// n=10, max_size=10 — negatives/zero/tiny/huge/duplicate + OOB clamps.
    /// No exact-ECDF dependency; just direct, simple checks.
    #[test]
    fn cdf_small_max10_smalln() {
        let mut values = vec![-1e9, -5.0, -2.0, 0.0, 2.0, 5.0, 1e-10, 2e-10, 2e-10, 1e9];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = TDigest::new_with_size(10).merge_sorted(values.clone());

        assert_cdf_oob_clamps(
            "cdf_small_max10_smalln",
            &td,
            values[0] - 1.0,
            values[values.len() - 1] + 1.0,
        );

        // Local order around dupes/tiny vs larger
        let trio = td.estimate_cdf(&[1e-10, 2e-10, 5.0]);
        assert_strict_increasing("cdf trio", &trio);
        assert_all_in_unit_interval("cdf trio bounds", &trio);
    }

    /// n=100, max_size=10 — accuracy vs exact ECDF (KS/MAE).
    #[test]
    fn cdf_small_max10_medn_100() {
        // 100-point set with duplicates and extremes
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect();
        values[0] = -1e9;
        values[1] = -30.0; // duplicate
        values[98] = 1e-10;
        values[99] = 1e9;
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let exact = exact_ecdf_for_sorted(&values);
        let td = TDigest::new_with_size(10).merge_sorted(values.clone());
        let approx = td.estimate_cdf(&values);

        let (ks, mae) = ks_mae(&exact, &approx);
        assert!(ks < 0.035, "CDF KS too large: {:.6e}", ks);
        assert!(mae < 0.003, "CDF MAE too large: {:.6e}", mae);

        assert_all_in_unit_interval("cdf(100) bounds", &approx);
        assert_monotone_chain("cdf(100) monotone", &approx);
    }
}
