package gr.tdigest_rs;

import java.io.*;
import java.nio.file.*;
import java.util.Locale;

final class Natives {
  private static boolean loaded = false;

  static synchronized void load() {
    if (loaded) return;

    final String os  = System.getProperty("os.name").toLowerCase(Locale.ROOT);
    final String arc = System.getProperty("os.arch").toLowerCase(Locale.ROOT);

    final String plat = os.contains("win") ? "windows"
                     : os.contains("mac") ? "macos"
                     : "linux";

    final String arch = (arc.equals("x86_64") || arc.equals("amd64")) ? "x86_64"
                     : (arc.equals("aarch64") || arc.equals("arm64")) ? "aarch64"
                     : (arc.equals("x86") || arc.equals("i386") || arc.equals("i686")) ? "x86"
                     : arc;

    final String lib = plat.equals("windows") ? "tdigest_rs.dll"
                   : plat.equals("macos")   ? "libtdigest_rs.dylib"
                                            : "libtdigest_rs.so";

    final String resource = "/META-INF/native/" + plat + "-" + arch + "/" + lib;

    // 0) Optional explicit override: -Dtdigest.rs.native=/abs/path/to/libtdigest_rs.so
    final String overridePath = System.getProperty("tdigest.rs.native");
    if (overridePath != null && !overridePath.isEmpty()) {
      System.load(overridePath);
      loaded = true;
      return;
    }

    // 1) Preferred: load from JAR resource (production packaging path)
    try (InputStream in = Natives.class.getResourceAsStream(resource)) {
      if (in != null) {
        Path tmp = Files.createTempFile("tdigest_rs-", lib);
        tmp.toFile().deleteOnExit();
        Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
        System.load(tmp.toAbsolutePath().toString());
        loaded = true;
        return;
      }
    } catch (IOException e) {
      // fall through to other strategies
    }

    // 2) Dev fallback: -Djava.library.path=... so loadLibrary can find it
    try {
      System.loadLibrary("tdigest_rs");
      loaded = true;
      return;
    } catch (UnsatisfiedLinkError ignored) {
      // keep falling through
    }

    // 3) Last-ditch: common local build path (useful when running from classes dir)
    try {
      Path local = Paths.get("target", "release", lib);
      if (Files.exists(local)) {
        System.load(local.toAbsolutePath().toString());
        loaded = true;
        return;
      }
    } catch (UnsatisfiedLinkError ignored) {
      // give a richer error below
    }

    throw new UnsatisfiedLinkError(
        "Failed loading native tdigest_rs: no resource " + resource +
        " on classpath, and loadLibrary/load() fallbacks failed. " +
        "Tip: run with -Djava.library.path=target/release or " +
        "-Dtdigest.rs.native=/abs/path/to/" + lib);
  }

  private Natives() {}
}
