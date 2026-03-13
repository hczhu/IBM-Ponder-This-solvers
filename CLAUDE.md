# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo contains solvers for [IBM Ponder This](https://research.ibm.com/haifa/ponderthis/index.shtml) monthly puzzles. Files are named `YYYY-MM` (e.g., `2026-01.cc`, `2025-11.py`). Each file is self-contained and includes both the solution logic and inline tests (GoogleTest for C++, unittest for Python).

## Build and Run

### C++ Solvers

```bash
# Build all C++ solvers
make

# Build a specific solver
make 2026-01.bin

# Run all tests (currently runs 2025-12.bin and 2025-11.py)
make test

# Clean binaries
make clean
```

**Dependencies:** `glog`, `gflags`, `gtest`, `fmt`, `libevent`, `libpthread`, `libbenchmark` (for `prime_number_gen_test.bin`), OpenMP.

**Compilation flags:** `-O3 -std=gnu++20 -fopenmp -DNDEBUG`

To run a C++ solver in solve mode (after tests pass):
```bash
./2026-01.bin solve
```

To run just the tests embedded in a C++ binary:
```bash
./2026-01.bin
```

### Python Solvers

```bash
# Run tests (no arguments)
python3 2025-11.py

# Run the solver
python3 2025-11.py solve
```

## Architecture

### C++ Solver Pattern

Each `.cc` file follows this structure:
1. A comment block at the top linking to the puzzle and describing the problem.
2. Core algorithm functions (`bruteForce`, `solve`, helper functions).
3. GoogleTest `TEST(...)` cases — including correctness tests that compare `bruteForce` vs `solve` on small inputs, and known puzzle examples.
4. A `main()` that calls `google::InitGoogleLogging`, `gflags::ParseCommandLineFlags`, runs all GoogleTest tests, then conditionally runs the real solver if `argv[1] == "solve"`.

### Python Solver Pattern

Each `.py` file uses `unittest` for tests. Running with no args runs the tests; passing `"solve"` as an argument (or via `if __name__ == "__main__"`) runs the full solver.

### Shared Library: `prime_number_gen`

`prime_number_gen.h` / `prime_number_gen.cc` is a reusable segmented Sieve of Eratosthenes:
- `PrimeNumberGen(low, high)`: sieves all primes in `[low, high]`.
- Supports range-for iteration via `begin()`/`end()`.
- Uses a compact bit-vector internally for memory efficiency.
- Solvers that use it (e.g., `2022-03.cc`, `2025-12.cc`) must be compiled with `prime_number_gen.cc` — see the Makefile for explicit rules.

### Multi-threading

Some C++ solvers (e.g., `2026-01.cc`) use `std::thread` with manual work partitioning by thread ID, and also use OpenMP (`-fopenmp`) to query available thread count at runtime.

### Coding styles
Follow the styles of the existing files for all programming languages.