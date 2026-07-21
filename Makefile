# Platform handling. On Linux with real GCC, -fopenmp and the system libraries
# work out of the box. Apple clang (the default g++ on macOS) needs extra help:
#   * OpenMP is provided by Homebrew's keg-only libomp via -Xclang -fopenmp.
#   * Homebrew headers/libs live under the native prefix (/opt/homebrew on Apple
#     Silicon) which is not on clang's default search path.
#   * glog >= 0.7 requires consumers to define GLOG_USE_GLOG_EXPORT (and, for
#     gflags integration, GLOG_USE_GFLAGS).
IS_CLANG := $(shell g++ --version 2>/dev/null | grep -ci clang)
ifeq ($(IS_CLANG),0)
  # Real GCC (typically Linux): -g is harmless and produces inline DWARF.
  PLATFORM_CXXFLAGS := -g
  PLATFORM_LDFLAGS :=
  OMP_CXXFLAGS := -fopenmp
  OMP_LDFLAGS :=
else
  # Apple clang: a single-step `-g` compile+link auto-invokes dsymutil to build a
  # <binary>.dSYM bundle. That step fails if a stale, differently-owned bundle
  # already exists, and the bundle is not needed for these solvers, so omit -g.
  # Prefer the native arm64 Homebrew at /opt/homebrew; fall back to `brew --prefix`.
  BREW_PREFIX := $(shell [ -x /opt/homebrew/bin/brew ] && echo /opt/homebrew || brew --prefix)
  OMP_PREFIX := $(BREW_PREFIX)/opt/libomp
  # -Wno-character-conversion silences a clang-only char8_t->char32_t warning
  # emitted from gtest's headers (gtest-printers.h); it is not our code.
  PLATFORM_CXXFLAGS := -I$(BREW_PREFIX)/include -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS -Wno-character-conversion
  PLATFORM_LDFLAGS := -L$(BREW_PREFIX)/lib
  OMP_CXXFLAGS := -Xclang -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS := -L$(OMP_PREFIX)/lib -lomp
endif

CXXFLAGS=-std=gnu++20 -Wall -Wno-deprecated -Wdeprecated-declarations -Wno-error=deprecated-declarations -Wno-sign-compare -Wno-unused -Wunused-label -Wunused-result -Wnon-virtual-dtor $(PLATFORM_CXXFLAGS) $(OMP_CXXFLAGS) -DNDEBUG
LDLIBS=$(PLATFORM_LDFLAGS) -lglog -levent -lgflags -lpthread -lgtest -lfmt $(OMP_LDFLAGS)

GCC_FLAGS=$(CXXFLAGS)
CPP_LIBS=$(LDLIBS)

SRCS_CC := $(filter-out prime_number_gen.cc prime_number_gen_test.cc, $(wildcard *.cc))
BINS := $(SRCS_CC:.cc=.bin) prime_number_gen_test.bin

all: $(BINS)

# Special rule for 2022-03.bin: this binary requires both 2022-03.cc and prime_number_gen.cc to be compiled and linked together.
# The generic pattern rule does not handle this dependency, so we specify it explicitly here.
2022-03.bin: 2022-03.cc prime_number_gen.cc
	g++ $^ -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

# Specific rule for 2025-12.bin which depends on prime_number_gen.cc
2025-12.bin: 2025-12.cc prime_number_gen.cc
	g++ $^ -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

# Specific rule for prime_number_gen_test.bin
prime_number_gen_test.bin: prime_number_gen_test.cc prime_number_gen.cc
	g++ $^ -O3 $(GCC_FLAGS) $(PLATFORM_LDFLAGS) -lglog -lgflags -lpthread -lgtest -lfmt -lbenchmark $(OMP_LDFLAGS) -o $@

%.bin: %.cc
	g++ $< -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

clean:
	rm -f *.bin

# 2026-03.bin requires -lbenchmark for the HopcroftKarp benchmarks
2026-03.bin: 2026-03.cc
	g++ $< -O3 $(GCC_FLAGS) $(CPP_LIBS) -lbenchmark -o $@

test: 2025-12.bin 2026-03.bin
	./2025-12.bin
	./2026-03.bin
	python3 2025-11.py

