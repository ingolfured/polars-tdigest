use crate::tdigest::TDigest;

impl TDigest {
    /// Estimate the value located at quantile `q` (0..=1).
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        if self.centroids().is_empty() {
            return 0.0;
        }
        if self.centroids().len() == 1 {
            return self.centroids()[0].mean();
        }

        let q = q.clamp(0.0, 1.0);
        let (target_index, total_weight) = self.quantile_to_weight_index(q);

        // Strict edge clamps: keep boundary cases (index==1 or index==N-1) in the interior logic.
        if target_index < 1.0 {
            return self.min();
        }
        if target_index > total_weight - 1.0 {
            return self.max();
        }

        let (left_idx, right_idx, left_center_cum_w, center_span_w) =
            self.find_bracketing_centroids(target_index);

        self.interpolate_between_centroids(
            left_idx,
            right_idx,
            left_center_cum_w,
            center_span_w,
            target_index,
        )
    }

    /// Map q∈[0,1] to a cumulative weight index in [0, total_weight].
    #[inline]
    fn quantile_to_weight_index(&self, q: f64) -> (f64, f64) {
        let total_weight = self.count();
        (q * total_weight, total_weight)
    }

    /// Half-weight bracketing: find adjacent centroids whose center-to-center span contains `index`.
    /// Returns (left_idx, right_idx, cumulative_weight_at_left_center, center_span_weight).
    fn find_bracketing_centroids(&self, index: f64) -> (usize, usize, f64, f64) {
        let first_w = self.centroids()[0].weight();
        let mut cum_w_at_left_center = first_w / 2.0;

        for left_idx in 0..(self.centroids().len() - 1) {
            let w_left = self.centroids()[left_idx].weight();
            let w_right = self.centroids()[left_idx + 1].weight();
            let center_span_weight = (w_left + w_right) / 2.0;

            if cum_w_at_left_center + center_span_weight > index {
                return (
                    left_idx,
                    left_idx + 1,
                    cum_w_at_left_center,
                    center_span_weight,
                );
            }
            cum_w_at_left_center += center_span_weight;
        }

        // Fallback: last pair (shouldn't happen in normal flow).
        let m = self.centroids().len();
        let w_last = self.centroids()[m - 1].weight();
        let span_w = (self.centroids()[m - 2].weight() + w_last) / 2.0;
        (m - 2, m - 1, self.count() - w_last / 2.0, span_w)
    }

    /// Interpolate between bracketing centroids with symmetric singleton/pile logic.
    fn interpolate_between_centroids(
        &self,
        left_idx: usize,
        right_idx: usize,
        cum_w_at_left_center: f64,
        center_span_weight: f64,
        target_index: f64,
    ) -> f64 {
        let left = &self.centroids()[left_idx];
        let right = &self.centroids()[right_idx];

        let (w_left, w_right) = (left.weight(), right.weight());
        let (m_left, m_right) = (left.mean(), right.mean());
        let right_center_cum_w = cum_w_at_left_center + center_span_weight;

        if left.is_singleton()
            && w_left > 1.0
            && Self::inside_pile_strict(target_index, cum_w_at_left_center, w_left)
        {
            return m_left;
        }
        if right.is_singleton()
            && w_right > 1.0
            && Self::inside_pile_strict(target_index, right_center_cum_w, w_right)
        {
            return m_right;
        }

        if w_left == 1.0 && (target_index - cum_w_at_left_center) < 0.5 {
            return m_left;
        }
        if w_right == 1.0 && (right_center_cum_w - target_index) < 0.5 {
            return m_right;
        }

        let dead_left = if w_left == 1.0 { 0.5 } else { 0.0 };
        let dead_right = if w_right == 1.0 { 0.5 } else { 0.0 };
        let weight_toward_right = target_index - cum_w_at_left_center - dead_left;
        let weight_toward_left = right_center_cum_w - target_index - dead_right;
        let denom = weight_toward_right + weight_toward_left;

        if denom <= 0.0 {
            0.5 * (m_left + m_right)
        } else {
            (m_left * weight_toward_left + m_right * weight_toward_right) / denom
        }
    }

    #[inline]
    fn inside_pile_strict(target_index: f64, center_cum_w: f64, pile_weight: f64) -> bool {
        // A “pile” is a centroid representing multiple identical values (singleton && weight>1).
        if pile_weight <= 1.0 {
            return false;
        }
        let half = pile_weight / 2.0;
        (center_cum_w - half) < target_index && target_index < (center_cum_w + half)
    }

    /// Return the indices of the two centroids that bracket the median (q = 0.5)
    /// using the same half-weight bracketing as estimate_quantile.
    fn bracket_centroids_for_median(&self) -> (usize, usize) {
        debug_assert!(!self.centroids().is_empty());
        if self.centroids().len() == 1 {
            return (0, 0);
        }
        let total_w = self.count();
        if self.centroids().len() == 2 || total_w <= 2.0 {
            return (0, 1);
        }
        let index = 0.5 * total_w;

        if index < 1.0 {
            return (0, 1);
        }
        if index > total_w - 1.0 {
            let m = self.centroids().len();
            return (m - 2, m - 1);
        }
        let (li, ri, _cw, _span) = self.find_bracketing_centroids(index);
        (li, ri)
    }

    /// Median with an even-count special case to avoid over-interpolation.
    /// - If total count is even: average the two neighboring centroid means around q=0.5.
    /// - If odd: fall back to `estimate_quantile(0.5)`.
    pub fn estimate_median(&self) -> f64 {
        let total = self.count();
        if total <= 0.0 {
            return 0.0;
        }
        if (total as i64) % 2 != 0 {
            return self.estimate_quantile(0.5);
        }
        let (li, ri) = self.bracket_centroids_for_median();
        let (ml, mr) = (self.centroids()[li].mean(), self.centroids()[ri].mean());
        (ml + mr) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use crate::tdigest::tdigest::TDigestBuilder;
    use crate::tdigest::test_helpers::*;

    #[test]
    fn centroid_addition_regression_pr_1() {
        // https://github.com/MnO2/t-digest/pull/1
        let vals = vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let mut t = TDigestBuilder::new().max_size(10).build();
        for v in vals {
            t = t.merge_unsorted(vec![v]);
        }
        assert_rel_close("median", 1.0, t.estimate_quantile(0.5), 0.01);
        assert_rel_close("q=0.95", 2.0, t.estimate_quantile(0.95), 0.01);

        let means: Vec<f64> = t.centroids().iter().map(|c| c.mean()).collect();
        assert_eq!(
            means.len(),
            2,
            "expected exactly two centroids, got {:?}",
            means
        );
        assert!(
            means.contains(&1.0) && means.contains(&2.0),
            "expected centroid means [1.0, 2.0], got {:?}",
            means
        );
    }

    /// n=10, max_size=10 — edge clamps, median bracket, monotone grid.
    #[test]
    fn quantiles_small_max10_smalln() {
        let mut values = vec![-10.0, -1.0, 0.0, 0.0, 2e-10, 1.0, 2.0, 10.0, 1e9, -1e9];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigestBuilder::new().max_size(10).build().merge_sorted(values.clone());

        assert_exact("Q(0)", *values.first().unwrap(), t.estimate_quantile(0.0));
        assert_exact("Q(1)", *values.last().unwrap(), t.estimate_quantile(1.0));

        let (lo_m, hi_m, _, _) = bracket(&values, 0.5);
        let med = t.estimate_quantile(0.5);
        assert_in_bracket("median", med, lo_m, hi_m, 4, 5);

        let grid = [
            t.estimate_quantile(0.01),
            t.estimate_quantile(0.10),
            t.estimate_quantile(0.25),
            med,
            t.estimate_quantile(0.75),
            t.estimate_quantile(0.90),
            t.estimate_quantile(0.99),
        ];
        assert_monotone_chain("quantiles grid", &grid);
    }

    /// n=100, max_size=10 — quantile lies between bracketing order stats.
    #[test]
    fn quantiles_small_max10_medn_100() {
        let mut values: Vec<f64> = (-30..=69).map(|x| x as f64).collect(); // 100 values
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = TDigestBuilder::new().max_size(10).build().merge_sorted(values.clone());

        for &(q, label) in &[
            (0.01_f64, "Q(0.01)"),
            (0.10_f64, "Q(0.10)"),
            (0.25_f64, "Q(0.25)"),
            (0.50_f64, "Q(0.50)"),
            (0.75_f64, "Q(0.75)"),
            (0.90_f64, "Q(0.90)"),
            (0.99_f64, "Q(0.99)"),
        ] {
            let (lo, hi, i_lo, i_hi) = bracket(&values, q);
            let x = t.estimate_quantile(q);
            assert_in_bracket(label, x, lo, hi, i_lo, i_hi);
        }
    }

    #[test]
    fn median_between_centroids_even_count() {
        // With symmetric bracketing, Q(0.5) equals the average of the two middle piles (→ 0.0).
        for num in [1, 2, 3, 10, 20] {
            let mut t = TDigestBuilder::new().max_size(100).build();
            for _ in 0..num {
                t = t.merge_sorted(vec![-1.0]);
            }
            for _ in 0..num {
                t = t.merge_sorted(vec![1.0]);
            }

            assert_exact("Q(0.5)", 0.0, t.estimate_quantile(0.5));
            assert_exact("median()", 0.0, t.estimate_median());
        }
    }

    #[test]
    fn quantile_exact_with_enough_capacity() {
        use crate::tdigest::test_helpers::assert_exact;
        use rand::{rngs::StdRng, Rng, SeedableRng};

        const N: usize = 9_999;
        let mut v: Vec<f64> = {
            let mut r = StdRng::seed_from_u64(42);
            (0..N).map(|_| r.random_range(0..N as u64) as f64).collect()
        };
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let td = TDigestBuilder::new().max_size(N + 1).build().merge_sorted(v.clone());

        assert_exact("Q(0)", v[0], td.estimate_quantile(0.0));
        assert_exact("Q(1)", v[N - 1], td.estimate_quantile(1.0));
        for (i, &x) in v.iter().enumerate() {
            let q = (i as f64 + 0.5) / N as f64; // mid-rank → exact order stat
            assert_exact("Q(mid)", x, td.estimate_quantile(q));
        }
    }
}
