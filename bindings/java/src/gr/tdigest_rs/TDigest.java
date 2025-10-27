package gr.tdigest_rs;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * Java wrapper for the native Rust TDigest.
 *
 * Parity with Python naming:
 *   - TDigest.fromArray(values, max_size, scale, f32_mode, singleton_policy, edges)
 *     where:
 *       values: double[] or float[] (overloads provided)
 *       max_size: Integer (default 1000; must be > 0)
 *       scale: "quad" | "k1" | "k2" | "k3" (default "k2")
 *       f32_mode: Boolean (default false) — quantize centroids to f32 internally
 *       singleton_policy: "off" | "use" | "edges" (default "use")
 *       edges: Integer (default 3 when policy == "edges")
 *
 * Queries:
 *   - cdf(double[] x) -> double[]
 *   - quantile(double q) -> double
 *   - median() -> double
 *
 * Serialization:
 *   - toBytes() / fromBytes(byte[])
 *
 * Back-compat factories:
 *   - create(int maxSize, String scale)
 *   - fromValues(double[] values, int maxSize, String scale)
 *
 * Builder API (ergonomic, type-safe):
 *   try (TDigest d = TDigest.builder()
 *       .maxSize(200)
 *       .scale(TDigest.Scale.K2)
 *       .singletonPolicy(TDigest.SingletonPolicy.USE)   // or EDGES with keep(...)
 *       .precision(TDigest.Precision.F64)               // internal f32 sketch if F32
 *       .build(new double[]{1,2,3})) {
 *     double p50 = d.median();
 *   }
 *
 * Owns a native handle; frees it via Cleaner or explicit close().
 */
public final class TDigest implements AutoCloseable {

  /* ======================= Type-safe enums for builder ======================= */

  /** TDigest scale family. Matches Rust {@code ScaleFamily}. */
  public enum Scale {
    QUAD, K1, K2, K3;
    String asWire() { return name().toLowerCase(Locale.ROOT); }
  }

  /** Singleton handling policy. Matches Rust {@code SingletonPolicy}. */
  public enum SingletonPolicy {
    OFF, USE, EDGES
  }

  /**
   * Internal centroid precision mode.
   * <p>External API stays {@code double[]} for queries; this only affects internal sketch storage.
   */
  public enum Precision {
    F64, F32
  }

  /* ======================= Backing native ======================= */

  private static String norm(String s) {
    if (s == null) throw new NullPointerException();
    return s.trim().toLowerCase(Locale.ROOT);
  }

  private static String parseScale(String s) {
    if (s == null) return "k2";
    String v = norm(s);
    switch (v) {
      case "quad":
      case "k1":
      case "k2":
      case "k3":
        return v;
      default:
        throw new IllegalArgumentException(
            "invalid scale: " + s + " (expected 'quad', 'k1', 'k2', or 'k3')");
    }
  }

  // Policy codes are passed to JNI as ints (0=Off, 1=Use, 2=Edges)
  private static int parsePolicyCode(String kind) {
    if (kind == null) return 1; // default "use"
    String v = norm(kind);
    switch (v) {
      case "off":   return 0;
      case "use":   return 1;
      case "edges":
      case "usewithprotectededges":
        return 2;
      default:
        throw new IllegalArgumentException(
            "invalid singleton_policy: " + kind + " (expected 'off', 'use', or 'edges')");
    }
  }

  private static int coalesceEdges(Integer edges) {
    return edges == null ? 3 : Math.max(0, edges);
  }

  private static int policyCode(SingletonPolicy p) {
    switch (p) {
      case OFF:  return 0;
      case USE:  return 1;
      case EDGES:return 2;
      default:   return 1;
    }
  }

  /* ======================= Cleaner & state ======================= */

  /** Single global cleaner used to free native memory when forgotten. */
  private static final Cleaner CLEANER = Cleaner.create();

  /** Internal holder for the native pointer. */
  private static final class State implements Runnable {
    private long handle;
    State(long handle) { this.handle = handle; }

    @Override
    public void run() {
      if (handle != 0) {
        try {
          TDigestNative.free(handle);
        } finally {
          handle = 0;
        }
      }
    }
  }

  private State state;                    // native handle state
  private final Cleaner.Cleanable cleanable;

  private TDigest(long handle) {
    if (handle == 0) {
      throw new IllegalStateException("Failed to create TDigest (null native handle)");
    }
    this.state = new State(handle);
    this.cleanable = CLEANER.register(this, state);
  }

  /* ======================= Builder (new, type-safe) ======================= */

  /**
   * Fluent builder for {@link TDigest}.
   * <p>Defaults: {@code maxSize=1000, scale=K2, policy=USE, keep=3 (if EDGES), precision=F64}.
   */
  public static final class Builder {
    private int maxSize = 1000;
    private Scale scale = Scale.K2;
    private SingletonPolicy policy = SingletonPolicy.USE;
    private int keep = 3;               // only used when policy==EDGES
    private Precision precision = Precision.F64;

    /** Set maximum centroid count; must be &gt; 0. */
    public Builder maxSize(int n) {
      if (n <= 0) throw new IllegalArgumentException("maxSize must be > 0");
      this.maxSize = n; return this;
    }

    /** Choose scale family (QUAD, K1, K2, K3). */
    public Builder scale(Scale s) {
      this.scale = Objects.requireNonNull(s, "scale"); return this;
    }

    /** Choose singleton policy (OFF, USE, EDGES). */
    public Builder singletonPolicy(SingletonPolicy p) {
      this.policy = Objects.requireNonNull(p, "policy"); return this;
    }

    /**
     * Number of edge singletons to protect when {@link SingletonPolicy#EDGES}.
     * Ignored otherwise. Negative values are clamped to 0.
     */
    public Builder keep(int k) {
      this.keep = Math.max(0, k); return this;
    }

    /** Internal centroid storage precision (F64 or F32). */
    public Builder precision(Precision p) {
      this.precision = Objects.requireNonNull(p, "precision"); return this;
    }

    /** Build from {@code double[]} input. */
    public TDigest build(double[] values) {
      Objects.requireNonNull(values, "values");
      long h = TDigestNative.fromArray(
          values, maxSize, scale.asWire(), policyCode(policy), keep, precision == Precision.F32);
      return new TDigest(h);
    }

    /** Build from {@code float[]} input (values are upcast to double for ingestion). */
    public TDigest build(float[] values) {
      Objects.requireNonNull(values, "values");
      long h = TDigestNative.fromArray(
          values, maxSize, scale.asWire(), policyCode(policy), keep, precision == Precision.F32);
      return new TDigest(h);
    }
  }

  /** Start a fluent builder with sensible defaults. */
  public static Builder builder() { return new Builder(); }

  /* ======================= Construction (stringly-typed, back-compat) ======================= */

  /**
   * Build from a double[] array-like.
   *
   * @param values               values (required)
   * @param max_size         default 1000; must be > 0
   * @param scale            default "k2"; one of "quad", "k1", "k2", "k3"
   * @param f32_mode         default false; if true, quantize centroids internally to f32
   * @param singleton_policy default "use"; one of "off", "use", "edges"
   * @param edges            default 3 when policy == "edges"; otherwise ignored
   */
  public static TDigest fromArray(
      double[] values,
      Integer max_size,
      String scale,
      Boolean f32_mode,
      String singleton_policy,
      Integer edges
  ) {
    Objects.requireNonNull(values, "values");
    final int maxSize = (max_size == null) ? 1000 : max_size;
    if (maxSize <= 0) throw new IllegalArgumentException("max_size must be > 0");

    final String sc = parseScale(scale);
    final int policyCode = parsePolicyCode(singleton_policy);
    final int keep = coalesceEdges(edges);
    final boolean f32 = (f32_mode != null) && f32_mode;

    long h = TDigestNative.fromArray(values, maxSize, sc, policyCode, keep, f32);
    return new TDigest(h);
  }

  /**
   * Build from a float[] array-like; same semantics as the double[] overload.
   */
  public static TDigest fromArray(
      float[] values,
      Integer max_size,
      String scale,
      Boolean f32_mode,
      String singleton_policy,
      Integer edges
  ) {
    Objects.requireNonNull(values, "values");
    final int maxSize = (max_size == null) ? 1000 : max_size;
    if (maxSize <= 0) throw new IllegalArgumentException("max_size must be > 0");

    final String sc = parseScale(scale);
    final int policyCode = parsePolicyCode(singleton_policy);
    final int keep = coalesceEdges(edges);
    final boolean f32 = (f32_mode != null) && f32_mode;

    long h = TDigestNative.fromArray(values, maxSize, sc, policyCode, keep, f32);
    return new TDigest(h);
  }

  /** Deserialize from bytes produced by {@link #toBytes()}. */
  public static TDigest fromBytes(byte[] bytes) {
    Objects.requireNonNull(bytes, "bytes");
    long h = TDigestNative.fromBytes(bytes);
    return new TDigest(h);
  }

  /* ======================= Back-compat factories ======================= */

  /** Create an empty digest (defaults: f32_mode=false, singleton_policy="use", edges=3). */
  public static TDigest create(int maxSize, String scale) {
    // Empty array → build from array; lets native create an empty digest and then merge nothing.
    return TDigest.fromArray(new double[0], maxSize, scale, false, "use", 3);
  }

  /** Build from values with legacy signature (routes to fromArray with defaults). */
  public static TDigest fromValues(double[] values, int maxSize, String scale) {
    return TDigest.fromArray(values, maxSize, scale, false, "use", 3);
  }

  /* ======================= Queries ======================= */

  /** CDF evaluated at array-like x (returns double[]). */
  public double[] cdf(double[] values) {
    ensureOpen();
    Objects.requireNonNull(values, "values");
    // copy defensively to avoid surprises from JNI pinning
    return TDigestNative.cdf(state.handle, Arrays.copyOf(values, values.length));
  }

  /** Quantile for q in [0,1]. */
  public double quantile(double q) {
    ensureOpen();
    return TDigestNative.quantile(state.handle, q);
  }

  /** Median (p = 0.5). */
  public double median() {
    ensureOpen();
    return TDigestNative.median(state.handle);
  }

  /* ======================= Serialization ======================= */

  /** Serialize digest to bytes (canonical bincode of the native TDigest). */
  public byte[] toBytes() {
    ensureOpen();
    return TDigestNative.toBytes(state.handle);
  }

  /* ======================= Lifecycle ======================= */

  /** Explicitly release the underlying native memory. */
  @Override
  public void close() {
    if (state != null && state.handle != 0) {
      cleanable.clean();   // runs State.run() exactly once
      state = null;
    }
  }

  /** True if already closed (or never opened). */
  public boolean isClosed() {
    return state == null || state.handle == 0;
  }

  private void ensureOpen() {
    if (isClosed()) {
      throw new IllegalStateException("TDigest is closed");
    }
  }

  @Override
  public String toString() {
    return "TDigest(handle=" + (state == null ? 0 : state.handle) + ")";
  }
}
