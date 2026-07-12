/*
  Source code for the puzzle
  https://research.ibm.com/blog/ponder-this-may-2026

  Given a square binary matrix A of order N over Z_2 with exactly one "1" in each
  row and column (i.e. a permutation matrix), find the lowest m > 0 such that
  A^m = I. This m is the order of the underlying permutation, which equals the
  least common multiple of its cycle lengths. The cycle lengths form a partition
  of N, so the smallest m with A^m = I is the LCM of the parts of that partition.

  g(N) is defined as the MAXIMUM such m over all N x N permutation matrices, i.e.

      g(N) = max { lcm(parts) : parts is a partition of N }.

  This is exactly Landau's function. The given examples confirm it:
      g(10)  = 30        = 2 * 3 * 5
      g(50)  = 180180    = 2^2 * 3^2 * 5 * 7 * 11 * 13

  The maximum is achieved by choosing, for a set of DISTINCT primes p, prime
  powers p^a whose sum is <= N and whose product (= the LCM) is maximal; any
  leftover units N - sum are fixed points (cycles of length 1) and do not change
  the LCM. So:

      g(N) = max { prod_i p_i^{a_i} : p_i distinct primes, sum_i p_i^{a_i} <= N }.

  Primary goal:  g(10^6) mod (10^9 + 7).
  Bonus goal:    g(10^8) mod (10^9 + 7).

  This is solved with a single exact grouped-knapsack DP over capacity N. For
  each prime we pick one exponent a >= 0 (cost p^a, benefit a*log p); maximizing
  the sum of benefits maximizes the product. Two facts make N = 10^8 tractable:

    1. Only primes up to ~sqrt(N log N) can ever appear in an optimum, so the
       prime count is ~ a few thousand, not pi(N).

    2. For a fixed prime p the DP dependencies db[c] -> db[c - p^a] all lie in
       the same residue class mod p, so the per-prime update splits into p
       INDEPENDENT descending chains and parallelizes perfectly across cores
       (OpenMP). Processing each chain top-down keeps db[c - p^a] at its
       pre-p value, giving the "at most one exponent of p" semantics with no
       double buffering.

  The DP carries the product modulo 10^9+7 alongside the (log) benefit, so no big
  integers or path reconstruction are needed. The optimal product is unique (it
  IS the quantity being maximized), so ties never cause ambiguity.

  g(10^6) runs in a few seconds. g(10^8) needs ~1.2 GB (one double + one uint32
  per cell) and is memory-latency bound: on a 12-core machine it takes roughly an
  hour (the bulk is ~8000 single-shift passes over the 10^8-cell array).
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using i64 = long long;

static constexpr i64 MOD = 1'000'000'007LL;
static constexpr double EPS = 1e-9;

// ---------------------------------------------------------------------------
// Primes
// ---------------------------------------------------------------------------
// Largest prime factor of g(N) is ~ sqrt(N log N); dpBound() for N=10^8 is
// ~65000, so a 200000 sieve is a comfortable ceiling.
static constexpr int SIEVE_MAX = 200000;
static std::vector<int> g_primes;       // all primes <= SIEVE_MAX
static std::vector<double> g_logp;      // log of the corresponding prime

static void initPrimes() {
  if (!g_primes.empty()) return;
  std::vector<bool> comp(SIEVE_MAX + 1, false);
  for (int i = 2; i <= SIEVE_MAX; ++i) {
    if (comp[i]) continue;
    g_primes.push_back(i);
    g_logp.push_back(std::log((double)i));
    for (i64 j = (i64)i * i; j <= SIEVE_MAX; j += i) comp[j] = true;
  }
}

static i64 powmod(i64 base, i64 exp) {
  i64 r = 1 % MOD;
  base %= MOD;
  while (exp > 0) {
    if (exp & 1) r = r * base % MOD;
    base = base * base % MOD;
    exp >>= 1;
  }
  return r;
}

// Integer p^a (a small, p^a <= N), no modulo.
static i64 ipow(i64 p, int a) {
  i64 r = 1;
  for (int i = 0; i < a; ++i) r *= p;
  return r;
}

// Cost of using prime p with exponent a: 0 if a==0 (prime unused), else p^a.
static inline i64 baseCost(i64 p, int a) { return a == 0 ? 0 : ipow(p, a); }

struct Res {
  double benefit;  // log of the product (the LCM)
  i64 mod;         // product modulo MOD
};

// ---------------------------------------------------------------------------
// Exact grouped-knapsack DP. Capacity = n. For each prime we choose one
// exponent a >= 0 (cost p^a, benefit a*log p); maximizing total benefit
// maximizes the product (the LCM). Considers only primes <= primeBound; the
// caller must pass a bound provably larger than the largest prime factor of
// g(n) (~ sqrt(n log n)).
//
// db[c] = best benefit with total cost <= c; pd[c] = that product mod MOD.
// For a fixed prime p the only dependencies are db[c] -> db[c - p^a], and
// c - p^a == c (mod p): every residue class mod p is an INDEPENDENT chain.
// Processing each chain top-down (descending c) means db[c - p^a], a lower
// index, still holds its pre-p value, giving the "at most one exponent of p"
// (0/1-knapsack) semantics with no double buffering. The classes partition
// [0, n] into disjoint cell sets, so the per-prime update parallelizes perfectly
// over residue classes (OpenMP). Product fits in uint32 since MOD < 2^30.
// ---------------------------------------------------------------------------
static Res solveDP(i64 n, i64 primeBound) {
  initPrimes();
  std::vector<double> db(n + 1, 0.0);
  std::vector<uint32_t> pd(n + 1, 1);

  for (size_t i = 0; i < g_primes.size(); ++i) {
    const i64 p = g_primes[i];
    LOG(INFO) << "Processing " << i << "-th prime " << p;
    if (p > primeBound || p > n) break;
    const double lp = g_logp[i];
#pragma omp parallel for schedule(dynamic, 64)
    for (i64 r = 0; r < p; ++r) {
      // Largest c <= n with c % p == r, then descend by p.
      for (i64 c = r + (n - r) / p * p; c >= p; c -= p) {
        double bb = db[c];
        uint32_t bp = pd[c];
        i64 pw = p;
        uint64_t pm = p % MOD;
        int a = 1;
        while (pw <= c) {
          const double cand = db[c - pw] + a * lp;
          if (cand > bb + EPS) {
            bb = cand;
            bp = (uint32_t)((uint64_t)pd[c - pw] * pm % MOD);
          }
          pw *= p;
          pm = pm * (uint64_t)(p % MOD) % MOD;
          ++a;
        }
        db[c] = bb;
        pd[c] = bp;
      }
    }
  }
  return {db[n], (i64)pd[n]};
}

// Safe prime bound for the exact DP at capacity n, found by summing primes
// 2+3+5+7+... until the running sum exceeds 2n, and returning that prime.
//
// Why this is an upper bound on the largest prime that can appear in g(n):
// summing primes up to x reaches ~ x^2/(2 ln x), so the prime where the sum
// passes 2n is x ~ sqrt(2 * 2n * ln x) ~ 1.41 * sqrt(n log n), which exceeds the
// proven bound P(g(n)) < 1.328 * sqrt(n log n) (Massias-Nicolas-Robin).
//
// The factor 2 (rather than 1) matters: in an optimal solution all primes up to
// the largest prime P would be used only in the LP/threshold relaxation; the
// integer optimum may instead skip a small prime to afford a larger one, so the
// honest "sum past n" point can fall just short of the true largest prime. Past
// 2n there is comfortable margin (and DPBoundSufficient checks it empirically).
static i64 dpBound(i64 n) {
  initPrimes();
  i64 sum = 0;
  for (int p : g_primes) {
    sum += p;
    if (sum > 2 * n) return p;
  }
  return g_primes.back();  // n exceeds the whole sieve; never happens here
}

// ---------------------------------------------------------------------------
// Brute force for tiny n: maximum lcm over all partitions of n.
// ---------------------------------------------------------------------------
static i64 bruteRec(int remaining, int maxPart, i64 curLcm, i64& best) {
  if (remaining == 0) {
    best = std::max(best, curLcm);
    return best;
  }
  for (int p = std::min(maxPart, remaining); p >= 1; --p) {
    bruteRec(remaining - p, p, std::lcm(curLcm, (i64)p), best);
  }
  return best;
}
static i64 bruteG(int n) {
  i64 best = 1;
  bruteRec(n, n, 1, best);
  return best;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
// Compare the DP against an independent brute force (max lcm over all
// partitions) for every n up to 48. This covers the parallel residue-class
// update with primes that have many classes (e.g. p up to 47), guarding against
// races and off-by-one errors.
TEST(Landau, BruteForce) {
  for (int n = 1; n <= 48; ++n) {
    const i64 g = bruteG(n);
    const Res r = solveDP(n, dpBound(n));
    EXPECT_NEAR(r.benefit, std::log((double)g), 1e-7) << "n=" << n;
    EXPECT_EQ(r.mod, g % MOD) << "n=" << n;
  }
}

TEST(Landau, GivenExamples) {
  // The two values stated in the puzzle.
  EXPECT_EQ(solveDP(10, dpBound(10)).mod, 30);
  EXPECT_EQ(solveDP(50, dpBound(50)).mod, 180180);
  EXPECT_NEAR(solveDP(50, dpBound(50)).benefit, std::log(180180.0), 1e-9);
}

TEST(Landau, DPBoundSufficient) {
  // The chosen prime bound must capture the optimum: doubling it (and, for the
  // larger cases, sieving more primes) must not change the answer.
  for (i64 n : {1000, 5000, 50000, 200000}) {
    const Res a = solveDP(n, dpBound(n));
    const Res b = solveDP(n, std::min(n, 2 * dpBound(n)));
    EXPECT_NEAR(a.benefit, b.benefit, 1e-6) << "n=" << n;
    EXPECT_EQ(a.mod, b.mod) << "n=" << n;
  }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  testing::InitGoogleTest(&argc, argv);
  const int res = RUN_ALL_TESTS();
  if (argc <= 1 || std::string(argv[1]) != "solve") {
    return res;
  }

  for (i64 n : {i64(1'000'000), i64(100'000'000)}) {
    const Res r = solveDP(n, dpBound(n));
    LOG(INFO) << "g(" << n << "): log benefit = " << r.benefit
              << " (primeBound = " << dpBound(n) << ")";
    std::cout << "g(" << n << ") mod (10^9+7) = " << r.mod << std::endl;
  }
  return res;
}
