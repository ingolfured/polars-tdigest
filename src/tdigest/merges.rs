use crate::tdigest::centroids::{coalesce_adjacent_equal_means, Centroid};
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Peekable;

/// Normalized stream returned by `normalize_stream`.
/// `out` is non-decreasing by mean, with *adjacent equal means coalesced* into a pile.
/// The totals support cheap updates of TDigest metadata.
pub struct NormalizedStream {
    pub out: Vec<Centroid>,
    pub total_w: f64,   // ∑ w
    pub total_mw: f64,  // ∑ (w * mean)
    pub min: f64,
    pub max: f64,
}

/// Normalize any centroid iterator into a clean, coalesced stream:
/// - verify non-decreasing means (panics with a clear message if violated),
/// - coalesce adjacent equal means into a single *singleton pile*,
/// - accumulate (∑w, ∑w·mean, min, max).
///
/// This is the *sole* owner of “equal means ⇒ pile” semantics for producers that
/// do not already coalesce internally.
pub fn normalize_stream<I>(items: I) -> NormalizedStream
where
    I: IntoIterator<Item = Centroid>,
{
    let mut out: Vec<Centroid> = Vec::new();
    let mut total_w = 0.0_f64;
    let mut total_mw = 0.0_f64;
    let (mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut prev_mean = f64::NEG_INFINITY;

    for c in items {
        let m = c.mean();
        let w = c.weight();

        if m < prev_mean {
            // Keep legacy substring so existing tests/assertions remain valid.
            panic!(
                "compress_into requires non-decreasing means; saw {} after {}",
                m, prev_mean
            );
        }

        if min_v == f64::INFINITY {
            min_v = m;
        }
        max_v = m;

        total_w += w;
        total_mw += w * m;

        out.push(c);
        prev_mean = m;
    }

    // Coalesce adjacent equals into a pile (singleton=true).
    let out = coalesce_adjacent_equal_means(out);

    NormalizedStream {
        out,
        total_w,
        total_mw,
        min: min_v,
        max: max_v,
    }
}

/// Merge stream that interleaves existing centroids with a run-length
/// encoding of sorted raw values (grouping identical values into one centroid).
///
/// IMPORTANT:
/// - This iterator focuses on correct *ordering* and optional value run-length encoding.
/// - It does **not** fuse values with existing centroids here; `normalize_stream` or downstream
///   coalescers own adjacent-equals fusion semantics.
pub(crate) struct MergeByMean<'a> {
    centroids: Peekable<std::slice::Iter<'a, Centroid>>,
    values: Peekable<std::slice::Iter<'a, f64>>,
    /// When pulling from values, group consecutive identical values into a single centroid.
    pending_value_run: Option<(f64, usize)>,
    singletons_on_values: bool,
}

impl<'a> MergeByMean<'a> {
    pub(crate) fn from_centroids_and_values(
        centroids: &'a [Centroid],
        values: &'a [f64],
        singletons_on_values: bool,
    ) -> Self {
        Self {
            centroids: centroids.iter().peekable(),
            values: values.iter().peekable(),
            pending_value_run: None,
            singletons_on_values,
        }
    }

    /// Drain a run of identical raw values into one centroid (pile).
    fn next_values_run(&mut self) -> Option<Centroid> {
        if let Some((val, len)) = self.pending_value_run.take() {
            return Some(if self.singletons_on_values {
                Centroid::new_singleton(val, len as f64)
            } else {
                Centroid::new_mixed(val, len as f64)
            });
        }
        let first = *self.values.next()?; // at least one value exists
        let mut len = 1usize;
        while let Some(&&v) = self.values.peek() {
            if v == first {
                self.values.next();
                len += 1;
            } else {
                break;
            }
        }
        Some(if self.singletons_on_values {
            Centroid::new_singleton(first, len as f64)
        } else {
            Centroid::new_mixed(first, len as f64)
        })
    }
}

impl<'a> Iterator for MergeByMean<'a> {
    type Item = Centroid;
    fn next(&mut self) -> Option<Self::Item> {
        // Always flush any stashed equal-mean value run before peeking again.
        if self.pending_value_run.is_some() {
            return self.next_values_run();
        }

        match (self.centroids.peek(), self.values.peek()) {
            (Some(c), Some(v)) => {
                if c.mean() < **v {
                    Some(*self.centroids.next().unwrap())
                } else if c.mean() > **v {
                    self.next_values_run()
                } else {
                    // Equal means: emit the existing centroid now, and stash the *whole* values run
                    // for the next call. We do NOT fuse here; normalization coalesces later.
                    let target = **v;
                    let mut len = 0usize;
                    while let Some(&&vv) = self.values.peek() {
                        if vv == target {
                            self.values.next();
                            len += 1;
                        } else {
                            break;
                        }
                    }
                    self.pending_value_run = Some((target, len));
                    Some(*self.centroids.next().unwrap())
                }
            }
            (Some(_), None) => Some(*self.centroids.next().unwrap()),
            (None, Some(_)) => self.next_values_run(),
            (None, None) => None,
        }
    }
}

/// k-way merge of centroid runs (by increasing mean), **with coalescing of equal means**.
/// If multiple runs have the same mean at their head, they are merged into a single centroid:
/// - mean: that value,
/// - weight: sum of weights,
/// - singleton: true iff all contributing centroids were singleton.
///
/// Rationale: historical behavior fused equal means at merge time for digest–digest merges,
/// which stabilizes mass layout and improves tail quantiles in uniform tests.
pub(crate) struct KWayCentroidMerge<'a> {
    runs: Vec<&'a [Centroid]>,
    pos: Vec<usize>,
    heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)>, // (mean, run_idx)
}

impl<'a> KWayCentroidMerge<'a> {
    pub(crate) fn new(runs: Vec<&'a [Centroid]>) -> Self {
        let mut heap = BinaryHeap::with_capacity(runs.len());
        let pos = vec![0; runs.len()];
        for (i, r) in runs.iter().enumerate() {
            if let Some(c) = r.first() {
                heap.push((Reverse(OrderedFloat::from(c.mean())), i));
            }
        }
        Self { runs, pos, heap }
    }

    #[inline]
    fn push_next(&mut self, run_idx: usize) {
        let p = self.pos[run_idx];
        let r = self.runs[run_idx];
        if p < r.len() {
            let c = &r[p];
            self.heap
                .push((Reverse(OrderedFloat::from(c.mean())), run_idx));
        }
    }
}

impl<'a> Iterator for KWayCentroidMerge<'a> {
    type Item = Centroid;

    fn next(&mut self) -> Option<Self::Item> {
        // Empty? Done.
        let (Reverse(min_mean_ord), first_run) = self.heap.pop()?;
        let min_mean = min_mean_ord.into_inner();

        // Start a coalesced centroid at this mean.
        let mut sum_w = 0.0f64;
        let mut all_singleton = true;

        // Consume the head item we just popped.
        {
            let p = &mut self.pos[first_run];
            let c = &self.runs[first_run][*p];
            debug_assert!(c.mean() == min_mean);
            sum_w += c.weight();
            all_singleton &= c.is_singleton();
            *p += 1;
            self.push_next(first_run);
        }

        // Now consume any other runs that also have this same mean at their head.
        while let Some((Reverse(peek_mean_ord), run_idx)) = self.heap.peek().copied() {
            if peek_mean_ord.into_inner() != min_mean {
                break;
            }
            let _ = self.heap.pop();
            let p = &mut self.pos[run_idx];
            let c = &self.runs[run_idx][*p];
            debug_assert!(c.mean() == min_mean);
            sum_w += c.weight();
            all_singleton &= c.is_singleton();
            *p += 1;
            self.push_next(run_idx);
        }

        // Emit a single centroid at `min_mean`.
        let mut out = Centroid::new(min_mean, sum_w);
        if all_singleton {
            out.mark_singleton(true);
        }
        Some(out)
    }
}
