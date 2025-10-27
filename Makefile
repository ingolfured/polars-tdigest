# ==============================================================================
# Makefile — tdigest-rs (Rust core + Python extension + Java/JNI)
# ==============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:
MAKEFLAGS += --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help

# ----------------------------------------------------------------------
# Pretty output helpers
# ----------------------------------------------------------------------
STYLE_OK    := $(shell tput setaf 2 2>/dev/null || printf '\033[32m')
STYLE_ERR   := $(shell tput setaf 1 2>/dev/null || printf '\033[31m')
STYLE_BOLD  := $(shell tput bold 2>/dev/null     || printf '\033[1m')
STYLE_RESET := $(shell tput sgr0 2>/dev/null     || printf '\033[0m')
STYLE_CODE  := $(shell tput setaf 6 2>/dev/null || printf '\033[36m')  # cyan

define banner
	@printf "\n$(STYLE_BOLD)==> %s$(STYLE_RESET)\n" "$(1)"
endef
define need
	@command -v $(1) >/dev/null 2>&1 || { printf "$(STYLE_ERR)✗ Missing dependency: $(1)$(STYLE_RESET)\n"; exit 1; }
	@printf "$(STYLE_OK)✓ $(1)$(STYLE_RESET)\n"
endef
define sep
	@printf "\n$(STYLE_BOLD)====================$(STYLE_RESET)\n"
endef

# ----------------------------------------------------------------------
# Tools & paths
# ----------------------------------------------------------------------
PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

CARGO  ?= cargo
UV     ?= uv
JAVAC  ?= javac
JAVA   ?= java
JAR    ?= jar
PYTHON ?= python3

# Python layout (single package tree inside bindings/python/)
PY_ROOT       := bindings/python
PY_PYPROJECT  := $(PY_ROOT)/pyproject.toml
PY_PKG_DIR    := $(PY_ROOT)/tdigest_rs
PY_TEST_DIR   := $(PY_ROOT)/tests

# CLI binary
LIB_DIR  := target/release
CLI_BIN  ?= tdigest
CLI_PATH := $(LIB_DIR)/$(CLI_BIN)

# Distributions
DIST ?= $(PY_ROOT)/dist

# Version (from Cargo.toml)
VER := $(shell sed -n 's/^version\s*=\s*"\(.*\)"/\1/p' Cargo.toml | head -1)

# Platform detection (for JNI JAR)
UNAME_S := $(shell uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M := $(shell uname -m)

# Python extension module naming (PyO3 / maturin)
EXPECTED_PYMODULE ?= tdigest_rs
EXPECTED_INITSYM  ?= PyInit_tdigest_rs
MATURIN_MODULE_NAME ?= tdigest_rs.$(EXPECTED_PYMODULE)

ifeq ($(findstring linux,$(UNAME_S)),linux)
  PLAT := linux
  LIBBASENAME := libtdigest_rs.so
else ifeq ($(findstring darwin,$(UNAME_S)),darwin)
  PLAT := macos
  LIBBASENAME := libtdigest_rs.dylib
else ifeq ($(findstring mingw,$(UNAME_S))$(findstring msys,$(UNAME_S)),mingwmsys)
  PLAT := windows
  LIBBASENAME := tdigest_rs.dll
else
  PLAT := linux
  LIBBASENAME := libtdigest_rs.so
endif

ifeq ($(UNAME_M),x86_64)
  ARCH := x86_64
else ifeq ($(UNAME_M),amd64)
  ARCH := x86_64
else ifeq ($(UNAME_M),aarch64)
  ARCH := aarch64
else ifeq ($(UNAME_M),arm64)
  ARCH := aarch64
else ifneq (,$(filter i386 i686 x86,$(UNAME_M)))
  ARCH := x86
else
  ARCH := $(UNAME_M)
endif

# Java
JAVA_SRC := bindings/java/src
CP_DEV   := target/java-classes

# ==============================================================================
# PHONY TARGETS
# ==============================================================================
.PHONY: help setup build fmt lint clean test help-me-run \
        rust-build rust-test rust-cli-smoke \
        py-build py-build-v py-test wheel \
        java-build java-test jar release

# ==============================================================================
# HELP
# ==============================================================================
help:
	@printf "\n$(STYLE_BOLD)Core$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "setup"        "Install toolchains and create a Python env with uv"
	@printf "  %-18s %s\n" "build"        "Build Rust lib+CLI (with smoke), Python extension, and Java classes"
	@printf "  %-18s %s\n" "release"      "Build CLI, Python wheel, and Java JARs — and smoke-test all three"
	@printf "  %-18s %s\n" "help-me-run"  "Four examples: CLI, Pure Python, Polars, Java (inline + run)"
	@printf "\n$(STYLE_BOLD)Dev-only$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "test"         "Run: Rust tests and Python tests"
	@printf "  %-18s %s\n" "fmt"          "Format Rust (rustfmt) and Python (ruff format)"
	@printf "  %-18s %s\n" "lint"         "Lint Rust (clippy -D warnings) and Python (ruff)"
	@printf "  %-18s %s\n" "clean"        "Remove Rust, Python, Java, and distribution artifacts"
	@printf "\n$(STYLE_BOLD)Rust$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "rust-build"   "cargo build --release (lib) + --bin $(CLI_BIN) (CLI), then smoke-test"
	@printf "  %-18s %s\n" "rust-test"    "cargo test -- --quiet"
	@printf "\n$(STYLE_BOLD)Python$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "py-build"     "maturin develop -r from $(PY_ROOT) (pyproject.toml lives there)"
	@printf "  %-18s %s\n" "py-test"      "pytest -q on $(PY_TEST_DIR)"
	@printf "  %-18s %s\n" "wheel"        "Build one wheel (manylinux_2_28+zig) from $(PY_ROOT), smoke-install"
	@printf "\n$(STYLE_BOLD)Java/JNI$(STYLE_RESET)\n"
	@printf "  %-18s %s\n" "java-build"   "Build native lib (features=java) and compile Java classes to $(CP_DEV)/"
	@printf "  %-18s %s\n" "jar"          "Package API JAR + native JAR; runs Java smoke after packaging"

# ==============================================================================
# Setup
# ==============================================================================
setup:
	$(call banner,Checking required host tools)
	$(call need,rustup)
	$(call need,cargo)
	$(call need,git)
	$(call need,uv)
	$(UV) --version

	$(call banner,Create .venv and install dev deps)
	$(UV) python install 3.12 || true
	$(UV) pip install "maturin>=1.9.5,<2.0" "ruff>=0.4" "pytest>=8.0" "polars>=1.34.0" numpy

	$(call banner,Quick Python import smoke)
	$(UV) run python -c "import polars as pl, sys; print('python', sys.version.split()[0], '| polars', pl.__version__, '| env ok')"

# ==============================================================================
# Core dev loop
# ==============================================================================
build:
	$(call banner,Build: Rust lib + CLI (with smoke))
	$(MAKE) rust-build
	$(call banner,Build: Python extension)
	$(MAKE) py-build
	$(call banner,Compile: Java)
	$(MAKE) java-build
	@printf "$(STYLE_OK)✓ all components built$(STYLE_RESET)\n"

fmt:
	$(CARGO) fmt --all
	$(UV) run ruff format .

lint:
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	$(UV) run ruff check .

clean:
	$(call banner,Clean: Rust target/)
	rm -rf target/
	$(CARGO) clean || true

	$(call banner,Clean: Java classes & jars)
	rm -rf "$(CP_DEV)" "target/java" "target/jar-staging"
	find "bindings/java" -type f \( -name '*.class' -o -name '*.java.bak' \) -delete || true

	$(call banner,Clean: Python artifacts)
	find bindings -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	find bindings -type f -name "*.so" -delete || true
	rm -rf .venv-wheeltest || true

	$(call banner,Clean: Distribution packages)
	rm -rf "$(DIST)" build/ *.egg-info/ || true

	@printf "$(STYLE_OK)✓ cleaned Rust, Python, Java, and distribution artifacts$(STYLE_RESET)\n"

# Only Rust + Python tests in aggregate
test: rust-test py-test
	@echo "✅ all tests passed"

# ==============================================================================
# Help me run — colored examples (printed only)
# ==============================================================================
help-me-run:
	$(call sep)
	@printf "$(STYLE_BOLD)1) Rust — CLI$(STYLE_RESET)\n"
	@printf "$(STYLE_CODE)echo '0 1 2 3' | target/release/$(CLI_BIN) --stdin --cmd quantile --p 0.5 --no-header --output csv$(STYLE_RESET)\n"
	@printf "# output: 0.5,1.5\n"

	$(call sep)
	@printf "$(STYLE_BOLD)2) Pure Python$(STYLE_RESET)\n"
	@printf "$(STYLE_CODE)uv run python -c \"import tdigest_rs as td; d=td.TDigest.from_array([0.0,1.0,2.0,3.0], max_size=100, scale='k2'); print('p50=', d.quantile(0.5)); print('cdf=', d.cdf([0.0,1.5,3.0]).tolist())\"$(STYLE_RESET)\n"

	$(call sep)
	@printf "$(STYLE_BOLD)3) Polars (lazy)$(STYLE_RESET)\n"
	@printf "$(STYLE_CODE)uv run python -c \"import polars as pl; from tdigest_rs.polars import tdigest, quantile; df=pl.DataFrame({'g':['a']*5,'x':[0,1,2,3,4]}); out=(df.lazy().group_by('g').agg(tdigest(pl.col('x'), max_size=100, scale='k2').alias('td')).select(quantile('td',0.5)).collect()); print(out)\"$(STYLE_RESET)\n"

	$(call sep)
	@printf "$(STYLE_BOLD)4) Java — inline Hello + compile + run$(STYLE_RESET)\n"
	@printf "# Build artifacts (once):\n"
	@printf "$(STYLE_CODE)make java-build jar$(STYLE_RESET)\n"
	@printf "$(STYLE_CODE)javac -cp target/java/tdigest-rs-java-$(VER).jar -d target/java-hello bindings/java/src/TestRun.java$(STYLE_RESET)\n"
	@printf "$(STYLE_CODE)java -Djava.library.path=$(LIB_DIR) -cp target/java/tdigest-rs-java-$(VER).jar:target/java-hello TestRun$(STYLE_RESET)\n"
	$(call sep)

# ==============================================================================
# Rust
# ==============================================================================
rust-build:
	$(CARGO) build --release
	$(CARGO) build --release --bin $(CLI_BIN)
	$(MAKE) rust-cli-smoke

rust-test:
	$(CARGO) test -- --quiet

Q ?= 0.5
$(CLI_PATH):
	$(CARGO) build --release --bin $(CLI_BIN)

rust-cli-smoke: $(CLI_PATH)
	@set -eu; \
	OUT="$$( echo '0 1 2 3' \
	  | '$(CLI_PATH)' --stdin --cmd quantile --p $(Q) --no-header --output csv \
	  | cut -d, -f2 )"; \
	printf "CLI p%.3g -> %s\n" "$(Q)" "$$OUT"; \
	[ "$$OUT" = "1.5" ] || { echo "❌ CLI quantile mismatch (got '$$OUT', want '1.5')"; exit 1; }; \
	echo "✅ rust_cli_smoke passed"

# ==============================================================================
# Python — single tree under bindings/python with its own pyproject.toml
# ==============================================================================
# We *cd* into $(PY_ROOT) so maturin reads THAT pyproject.toml.
# We pass --manifest-path back to the repo's Cargo.toml.

py-build:
	# Preconditions so failures are loud and obvious
	[ -f "$(PY_PYPROJECT)" ] || { echo "$(STYLE_ERR)✗ Missing $(PY_PYPROJECT)$(STYLE_RESET)"; exit 1; }
	[ -f "$(PY_PKG_DIR)/__init__.py" ] || { echo "$(STYLE_ERR)✗ Missing $(PY_PKG_DIR)/__init__.py$(STYLE_RESET)"; exit 1; }
	[ -f "$(PY_PKG_DIR)/polars/__init__.py" ] || { echo "$(STYLE_ERR)✗ Missing $(PY_PKG_DIR)/polars/__init__.py$(STYLE_RESET)"; exit 1; }
	grep -E 'from[[:space:]]+\.[[:space:]]*tdigest_rs[[:space:]]+import' "$(PY_PKG_DIR)/__init__.py" >/dev/null \
	  || { echo "$(STYLE_ERR)✗ __init__.py must import from .tdigest_rs (no underscore)$(STYLE_RESET)"; exit 1; }

	# Build into the active venv directly
	set -e; \
	cd "$(PY_ROOT)"; \
	if ! $(UV) run maturin develop -r --manifest-path ../../Cargo.toml -F python; then \
	  printf "$(STYLE_ERR)✗ maturin develop failed — retrying with verbose logs$(STYLE_RESET)\n"; \
	  MATURIN_LOG=debug $(UV) run maturin develop -r --manifest-path ../../Cargo.toml -F python -v -v; \
	fi

py-build-v:
	@set -e; cd "$(PY_ROOT)"; \
	UV_LOG=info MATURIN_LOG=info \
	$(UV) run maturin develop -r -F python \
		--manifest-path ../../Cargo.toml -v -v


py-test: py-build
	$(UV) run pytest -q $(PY_TEST_DIR)

wheel:
	@set -eu; \
	cd bindings/python; \
	rm -rf dist && mkdir -p dist; \
	uv run maturin build -r --manifest-path ../../Cargo.toml -F python \
		--compatibility manylinux_2_28 --zig -o dist; \
	WHEEL="$$(ls -1t dist/*.whl | head -1)"; \
	python3 -m venv .venv-wheel; \
	. .venv-wheel/bin/activate; \
	pip install -U pip >/dev/null; \
	pip install --no-deps "$$WHEEL" >/dev/null; \
	python -c "import tdigest_rs as td; d=td.TDigest.from_array([0,1,2,3],max_size=100,scale='k2'); print('cdf:', d.cdf([0,1.5,3]).tolist())"; \
	rm -rf .venv-wheel



# ==============================================================================
# Java / JNI
# ==============================================================================
java-build:
	$(call need,javac)
	$(call need,java)
	$(CARGO) build --release --no-default-features --features java --lib
	rm -rf "$(CP_DEV)"
	mkdir -p "$(CP_DEV)"
	javac -d "$(CP_DEV)" $(shell find "$(JAVA_SRC)" -type f -name '*.java')

java-test:
	@set -e; \
	OUT="$$( $(JAVA) -Djava.library.path=$(LIB_DIR) -cp $(CP_DEV) TestRun )"; \
	echo "• Java smoke output:"; \
	echo "$$OUT"; \
	ARR_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^\[[[:space:]0-9\.,]+\]$$' || true)"; \
	P50_LINE="$$(printf "%s\n" "$$OUT" | grep -m1 -E '^p50[[:space:]]*=[[:space:]]*1\.5$$' || true)"; \
	[ "$$ARR_LINE" = "[0.125, 0.5, 0.875]" ] || { echo "❌ array mismatch (classes)"; exit 1; }; \
	[ -n "$$P50_LINE" ] || { echo "❌ p50 line missing (classes)"; exit 1; }; \
	echo "✅ java_test (classes) passed"

jar: java-build
	$(call banner,Package Java API JAR)
	mkdir -p "target/java" "target/jar-staging"
	printf "Manifest-Version: 1.0\nAutomatic-Module-Name: gr.tdigest_rs\n" > "target/java/MANIFEST.MF"
	$(JAR) cfm "target/java/tdigest-rs-java-$(VER).jar" "target/java/MANIFEST.MF" -C "$(CP_DEV)" .
	@printf "$(STYLE_OK)✓ API JAR -> %s$(STYLE_RESET)\n" "target/java/tdigest-rs-java-$(VER).jar"

	$(call banner,Package platform native JAR)
	rm -rf "target/jar-staging/natives" && mkdir -p "target/jar-staging/natives/$(PLAT)-$(ARCH)"
	cp "$(LIB_DIR)/$(LIBBASENAME)" "target/jar-staging/natives/$(PLAT)-$(ARCH)/$(LIBBASENAME)"
	$(JAR) cf "target/java/tdigest-rs-java-$(VER)-$(PLAT)-$(ARCH).jar" -C "target/jar-staging" natives
	@printf "$(STYLE_OK)✓ Native JAR -> %s$(STYLE_RESET)\n" "target/java/tdigest-rs-java-$(VER)-$(PLAT)-$(ARCH).jar"

	$(call banner,Java smoke (post-jar))
	$(MAKE) java-test

# ==============================================================================
# One-shot RELEASE: build & smoke-test all artifacts
# ==============================================================================
release:
	$(call banner,Release: Build Rust CLI (with smoke))
	$(MAKE) rust-build
	@printf "• CLI -> %s\n" "$(CLI_PATH)"

	$(call banner,Release: Build Wheel (with smoke))
	$(MAKE) wheel
	LAST_WHEEL="$$(ls -1t "$(DIST)"/*.whl 2>/dev/null | head -1 || true)"; \
	if [ -z "$$LAST_WHEEL" ]; then \
		printf "$(STYLE_ERR)✗ No wheel found in %s$(STYLE_RESET)\n" "$(DIST)"; exit 1; \
	fi; \
	printf "• Wheel -> %s\n" "$$LAST_WHEEL"; \
	echo "$$LAST_WHEEL" > .last-wheel-path

	$(call banner,Release: Build JARs (with smoke))
	$(MAKE) jar
	@printf "• API JAR    -> %s\n" "target/java/tdigest-rs-java-$(VER).jar"
	@printf "• Native JAR -> %s\n" "target/java/tdigest-rs-java-$(VER)-$(PLAT)-$(ARCH).jar"

	@LAST_WHL="$$(cat .last-wheel-path 2>/dev/null || true)"; \
	rm -f .last-wheel-path; \
	printf "\n$(STYLE_BOLD)==> Artifacts$(STYLE_RESET)\n"; \
	printf "  CLI binary : %s\n" "$(CLI_PATH)"; \
	printf "  Wheel      : %s\n" "$${LAST_WHL:-<none>}"; \
	printf "  Native JAR : %s\n" "target/java/tdigest-rs-java-$(VER)-$(PLAT)-$(ARCH).jar"; \
	printf "\n$(STYLE_OK)✓ Release artifacts built & smoke-tested$(STYLE_RESET)\n"
