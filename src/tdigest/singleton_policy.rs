use serde::{Deserialize, Serialize};

/// Controls how TDigest treats true singletons (weight==1 or piles) and edges.
///
/// Semantics:
/// - `Off`: everything is treated uniformly. No tail protection, no special handling for
///          data-defined singletons or piles. Capacity applies to the *whole* digest.
/// - `Use`: data-defined singletons/piles are respected, but no tail protection. Capacity
///          applies to the *whole* digest, including edges.
/// - `UseWithProtectedEdges(k)`: like `Use`, but preserve up to `k` raw singletons (or piles
///          at the same mean) on *each* edge. Those edge items do not count toward the core
///          capacity; only the interior does.
///
/// Note: “singleton” here is a *data* fact: either a raw item (weight==1) or a run of identical
/// values fused into a single centroid (“pile”) with weight>1. Mixed clusters are everything else.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SingletonPolicy {
    /// Treat everything uniformly — no special casing for singletons or piles.
    Off,
    /// Respect data-defined singletons/piles, no tail protection.
    #[default]
    Use,
    /// Respect data-defined singletons/piles and preserve up to N of them at each edge,
    /// outside the core capacity.
    UseWithProtectedEdges(usize),
}
