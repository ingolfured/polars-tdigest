// src-java/gr/tdigest_rs/TDigestNative.java
package gr.tdigest_rs;

final class TDigestNative {
  static {
    System.loadLibrary("tdigest_rs"); // ensure libtdigest_rs.so/dylib/dll is on the path
  }
  static native long   fromBytes(byte[] bytes);
  static native byte[] toBytes(long handle);
  static native long   fromArray(Object values, int maxSize, String scale, int policyCode, int edges, boolean f32mode);
  static native void   free(long handle);
  static native double[] cdf(long handle, double[] values);
  static native double   quantile(long handle, double q);
  static native double   median(long handle);
  private TDigestNative() {}
}
