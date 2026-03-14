/*
  Source code for the puzzle
  https://research.ibm.com/blog/ponder-this-march-2026

  Alice and Bob play on an N×M board. Alice places a pawn on (x,y). Bob moves
  it to an adjacent (non-diagonal) square and removes the vacated square.
  Players alternate; the player who cannot move loses.

  Square (i,j) is labeled p^i + q^j (1-indexed). For each prime s in S, all
  squares whose label is divisible by s are removed before play begins.

  Each remaining square is classified "A" (Alice wins) or "B" (Bob wins).

  Game-theoretic reduction: after Alice places at (x,y), Bob moves first. This
  is exactly Vertex Geography on the remaining grid graph G_s. By the theorem
  of Fraenkel, Scheinerman & Ullman (1993), on a bipartite graph the first
  player wins iff the starting vertex is in every maximum matching.

  Therefore:
    "B" (Bob/first-mover wins) <=> (x,y) is in every maximum matching of G_s
    "A" (Alice wins)           <=> (x,y) is NOT in every maximum matching of G_s

  To classify all vertices simultaneously, use the Dulmage-Mendelsohn
  decomposition after running Hopcroft-Karp:

  Build directed graph D: matched edges R->L, unmatched edges L->R.
    - BFS from free L-nodes in D:   reachable L-nodes are "A"
    - BFS from free R-nodes in D^T: reachable R-nodes are "A"
  All other matched nodes are "B".

  The total answer is the sum across each prime s in S of the A and B counts
  on the board for that prime.
*/

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <omp.h>
#include <queue>
#include <span>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Sieve of Eratosthenes
// ---------------------------------------------------------------------------
std::vector<int> primesBelow(int limit) {
  std::vector<bool> composite(limit, false);
  std::vector<int> primes;
  for (int i = 2; i < limit; ++i) {
    if (!composite[i]) {
      primes.push_back(i);
      for (long long j = (long long)i * i; j < limit; j += i) {
        composite[j] = true;
      }
    }
  }
  return primes;
}

TEST(Sieve, Empty) {
  EXPECT_TRUE(primesBelow(0).empty());
  EXPECT_TRUE(primesBelow(1).empty());
  EXPECT_TRUE(primesBelow(2).empty());
}

TEST(Sieve, SmallValues) {
  EXPECT_EQ(primesBelow(3),  (std::vector<int>{2}));
  EXPECT_EQ(primesBelow(4),  (std::vector<int>{2, 3}));
  EXPECT_EQ(primesBelow(12), (std::vector<int>{2, 3, 5, 7, 11}));
}

TEST(Sieve, ExclusiveUpperBound) {
  // limit itself must not be included even when it is prime
  auto p10 = primesBelow(10);
  EXPECT_EQ(p10.back(), 7);
  auto p11 = primesBelow(11);
  EXPECT_EQ(p11.back(), 7);
  auto p12 = primesBelow(12);
  EXPECT_EQ(p12.back(), 11);
}

TEST(Sieve, Count100) {
  // There are 25 primes below 100
  EXPECT_EQ(primesBelow(100).size(), 25u);
  EXPECT_EQ(primesBelow(100).front(), 2);
  EXPECT_EQ(primesBelow(100).back(), 97);
}

TEST(Sieve, AllPrimesArePrime) {
  // Every returned value must actually be prime (no composite sneaks in)
  for (int p : primesBelow(200)) {
    for (int d = 2; d * d <= p; ++d) {
      EXPECT_NE(p % d, 0) << p << " is not prime";
    }
  }
}

TEST(Sieve, NoneSkipped) {
  // Every prime below limit must appear (no prime is missing)
  auto primes = primesBelow(200);
  std::vector<bool> inResult(200, false);
  for (int p : primes) { inResult[p] = true; }
  for (int n = 2; n < 200; ++n) {
    bool isPrime = true;
    for (int d = 2; d * d <= n; ++d) {
      if (n % d == 0) { isPrime = false; break; }
    }
    EXPECT_EQ(inResult[n], isPrime) << "n=" << n;
  }
}

// ---------------------------------------------------------------------------
// Hopcroft-Karp bipartite matching
//
// L-nodes: 0 .. nL-1
// R-nodes: 0 .. nR-1
//
// Edges are stored in flat arrays: each node has a fixed-size slot of max_k
// entries.  adj[l*max_k .. l*max_k+adj_count[l]) are the R-neighbors of l;
// radj[r*max_k .. r*max_k+radj_count[r]) are the L-neighbors of r.
// This avoids the pointer-chasing overhead of nested std::vector.
//
// Returns match_l[l] = r matched to l (-1 if unmatched)
//         match_r[r] = l matched to r (-1 if unmatched)
// ---------------------------------------------------------------------------
struct HopcroftKarp {
  int nL, nR, max_k;
  std::vector<int> adj,        adj_count;   // adj[l*max_k + i], adj_count[l]
  std::vector<int> radj,       radj_count;  // radj[r*max_k + i], radj_count[r]
  std::vector<int> match_l, match_r, dist;
  int bfs_depth = 0;
  // Reused across bfs() calls to avoid repeated heap allocation/deallocation.
  std::vector<int> bfs_cur, bfs_nxt, bfs_free_r;

  HopcroftKarp(int nL, int nR, int max_k)
      : nL(nL), nR(nR), max_k(max_k),
        adj(nL * max_k),  adj_count(nL, 0),
        radj(nR * max_k), radj_count(nR, 0),
        match_l(nL, -1), match_r(nR, -1), dist(nL) {}

  void addEdge(int l, int r) {
    adj [l * max_k + adj_count [l]++] = r;
    radj[r * max_k + radj_count[r]++] = l;
  }

  std::span<int> adj_of (int l) { return {adj.data()  + l * max_k, (size_t)adj_count [l]}; }
  std::span<int> radj_of(int r) { return {radj.data() + r * max_k, (size_t)radj_count[r]}; }

  // Level-by-level BFS from free L-nodes.  Returns all free R-nodes reachable
  // at the shortest augmenting-path distance (bfs_depth = the L-layer index at
  // which those R-nodes were discovered).  An empty return means no augmenting
  // path exists.
  const std::vector<int>& bfs() {
    bfs_cur.clear(); bfs_nxt.clear(); bfs_free_r.clear();
    for (int l = 0; l < nL; ++l) {
      if (match_l[l] == -1) { dist[l] = 0; bfs_cur.push_back(l); }
      else                    dist[l] = INT_MAX;
    }
    for (int d = 0; !bfs_cur.empty(); ++d) {    // first loop: L-layer distance
      bfs_nxt.clear();
      for (int l : bfs_cur) {                    // second loop: L-nodes at layer d
        for (int r : adj_of(l)) {
          int l2 = match_r[r];
          if (l2 == -1) {
            bfs_free_r.push_back(r);             // free R-node: end of augmenting path
          } else if (dist[l2] == INT_MAX) {
            dist[l2] = d + 1;
            bfs_nxt.push_back(l2);
          }
        }
      }
      if (!bfs_free_r.empty()) { bfs_depth = d; return bfs_free_r; }  // exit first loop
      std::swap(bfs_cur, bfs_nxt);
    }
    return bfs_free_r;  // empty: no augmenting paths
  }

  // DFS in the *reverse* layered graph starting from R-node r.
  // Traverses: unmatched R→L (via radj), then matched L→R (via match_l),
  // alternating until a free L-node (dist==0, match_l==-1) is reached.
  // L-nodes on the path must lie at successive dist layers d, d-1, ..., 0.
  // Marking dist[l]=INT_MAX immediately prevents any other DFS from entering l.
  bool dfs(int r, int d) {
    for (int l : radj_of(r)) {
      if (dist[l] != d) { continue; }
      dist[l] = INT_MAX;                    // mark l used; prevent revisiting
      if (match_l[l] == -1 || dfs(match_l[l], d - 1)) {
        match_l[l] = r;
        match_r[r] = l;
        return true;
      }
    }
    return false;
  }

  int maxMatching() {
    int res = 0;
    while (!bfs().empty()) {
      for (int r : bfs_free_r) {
        // match_r[r]==-1 guards against duplicate entries in bfs_free_r and
        // against a free R-node matched by an earlier dfs() this phase.
        if (match_r[r] == -1 && dfs(r, bfs_depth)) {
          ++res;
        }
      }
    }
    return res;
  }
};

// ---------------------------------------------------------------------------
// Build a HopcroftKarp graph from an N×M grid with the puzzle's removal rule
// for prime s, without running the matching.  Used by benchmarks and tests.
// Grid nodes have at most 4 neighbors, so max_k=4.
// ---------------------------------------------------------------------------
HopcroftKarp buildGridHK(int N, int M, int64_t p, int64_t q, int s) {
  std::vector<int> powP(N + 1), powQ(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * p % s; powP[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= M; ++j) { acc = acc * q % s; powQ[j] = (int)acc; }
  }
  int total = N * M;
  std::vector<int> lIdx(total, -1), rIdx(total, -1);
  int nL = 0, nR = 0;
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      if ((powP[i] + powQ[j]) % s == 0) { continue; }
      int f = (i - 1) * M + (j - 1);
      if ((i + j) % 2 == 0) { lIdx[f] = nL++; }
      else                   { rIdx[f] = nR++; }
    }
  }
  HopcroftKarp hk(nL, nR, /*max_k=*/4);
  const int di[] = {0, 0, 1, -1}, dj[] = {1, -1, 0, 0};
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      int f = (i - 1) * M + (j - 1);
      if (lIdx[f] == -1) { continue; }
      for (int d = 0; d < 4; ++d) {
        int ni = i + di[d], nj = j + dj[d];
        if (ni < 1 || ni > N || nj < 1 || nj > M) { continue; }
        int nf = (ni - 1) * M + (nj - 1);
        if (rIdx[nf] == -1) { continue; }
        hk.addEdge(lIdx[f], rIdx[nf]);
      }
    }
  }
  return hk;
}

// ===========================================================================
// HopcroftKarp Unit Tests  (placed here for locality with the implementation)
// ===========================================================================

TEST(HopcroftKarp, SingleEdge) {
  HopcroftKarp hk(1, 1, /*max_k=*/1);
  hk.addEdge(0, 0);
  EXPECT_EQ(hk.maxMatching(), 1);
  EXPECT_EQ(hk.match_l[0], 0);
  EXPECT_EQ(hk.match_r[0], 0);
}

TEST(HopcroftKarp, PathGraph) {
  // L={0,1}, R={0}: edges 0-0, 1-0. Max matching = 1.
  HopcroftKarp hk(2, 1, /*max_k=*/1);
  hk.addEdge(0, 0);
  hk.addEdge(1, 0);
  EXPECT_EQ(hk.maxMatching(), 1);
  int matched = (hk.match_l[0] != -1 ? 0 : 1);
  EXPECT_EQ(hk.match_l[matched], 0);
  EXPECT_EQ(hk.match_l[1 - matched], -1);
  EXPECT_EQ(hk.match_r[0], matched);
}

TEST(HopcroftKarp, StarFromLeft) {
  // L={0}, R={0,1,2,3}: L[0] connects all R. Max matching = 1.
  HopcroftKarp hk(1, 4, /*max_k=*/4);
  for (int r = 0; r < 4; ++r) { hk.addEdge(0, r); }
  EXPECT_EQ(hk.maxMatching(), 1);
  EXPECT_NE(hk.match_l[0], -1);
  int mr = hk.match_l[0];
  EXPECT_EQ(hk.match_r[mr], 0);
  for (int r = 0; r < 4; ++r) {
    if (r != mr) { EXPECT_EQ(hk.match_r[r], -1); }
  }
}

TEST(HopcroftKarp, StarFromRight) {
  // L={0,1,2,3}, R={0}: all L connect R[0]. Max matching = 1.
  HopcroftKarp hk(4, 1, /*max_k=*/1);
  for (int l = 0; l < 4; ++l) { hk.addEdge(l, 0); }
  EXPECT_EQ(hk.maxMatching(), 1);
  EXPECT_NE(hk.match_r[0], -1);
  int ml = hk.match_r[0];
  EXPECT_EQ(hk.match_l[ml], 0);
  for (int l = 0; l < 4; ++l) {
    if (l != ml) { EXPECT_EQ(hk.match_l[l], -1); }
  }
}

TEST(HopcroftKarp, PerfectMatching_K22) {
  HopcroftKarp hk(2, 2, /*max_k=*/2);
  hk.addEdge(0, 0); hk.addEdge(0, 1);
  hk.addEdge(1, 0); hk.addEdge(1, 1);
  EXPECT_EQ(hk.maxMatching(), 2);
  for (int l = 0; l < 2; ++l) { EXPECT_NE(hk.match_l[l], -1); }
  for (int r = 0; r < 2; ++r) { EXPECT_NE(hk.match_r[r], -1); }
}

TEST(HopcroftKarp, PerfectMatching_K33) {
  HopcroftKarp hk(3, 3, /*max_k=*/3);
  for (int l = 0; l < 3; ++l) {
    for (int r = 0; r < 3; ++r) {
      hk.addEdge(l, r);
    }
  }
  EXPECT_EQ(hk.maxMatching(), 3);
  for (int l = 0; l < 3; ++l) { EXPECT_NE(hk.match_l[l], -1); }
  for (int r = 0; r < 3; ++r) { EXPECT_NE(hk.match_r[r], -1); }
}

TEST(HopcroftKarp, EmptyGraph) {
  HopcroftKarp hk(3, 3, /*max_k=*/1);
  EXPECT_EQ(hk.maxMatching(), 0);
  for (int l = 0; l < 3; ++l) { EXPECT_EQ(hk.match_l[l], -1); }
  for (int r = 0; r < 3; ++r) { EXPECT_EQ(hk.match_r[r], -1); }
}

TEST(HopcroftKarp, DisconnectedComponents) {
  // Two disjoint K_{2,2}: max matching = 4.
  HopcroftKarp hk(4, 4, /*max_k=*/2);
  hk.addEdge(0, 0); hk.addEdge(0, 1);
  hk.addEdge(1, 0); hk.addEdge(1, 1);
  hk.addEdge(2, 2); hk.addEdge(2, 3);
  hk.addEdge(3, 2); hk.addEdge(3, 3);
  EXPECT_EQ(hk.maxMatching(), 4);
  for (int l = 0; l < 4; ++l) { EXPECT_NE(hk.match_l[l], -1); }
  for (int r = 0; r < 4; ++r) { EXPECT_NE(hk.match_r[r], -1); }
}

TEST(HopcroftKarp, AugmentingPathChain) {
  // Chain requiring multiple BFS rounds:
  // L={0,1,2}, R={0,1,2}. Edges: 0-0, 1-0, 1-1, 2-1, 2-2. Max matching = 3.
  HopcroftKarp hk(3, 3, /*max_k=*/2);
  hk.addEdge(0, 0);
  hk.addEdge(1, 0); hk.addEdge(1, 1);
  hk.addEdge(2, 1); hk.addEdge(2, 2);
  EXPECT_EQ(hk.maxMatching(), 3);
  for (int l = 0; l < 3; ++l) { EXPECT_NE(hk.match_l[l], -1); }
  for (int r = 0; r < 3; ++r) { EXPECT_NE(hk.match_r[r], -1); }
}

TEST(HopcroftKarp, AsymmetricSizes_K34) {
  // K_{3,4}: max matching = 3 (limited by left side).
  HopcroftKarp hk(3, 4, /*max_k=*/4);
  for (int l = 0; l < 3; ++l) {
    for (int r = 0; r < 4; ++r) {
      hk.addEdge(l, r);
    }
  }
  EXPECT_EQ(hk.maxMatching(), 3);
  for (int l = 0; l < 3; ++l) { EXPECT_NE(hk.match_l[l], -1); }
  int matchedR = 0;
  for (int r = 0; r < 4; ++r) {
    if (hk.match_r[r] != -1) { ++matchedR; }
  }
  EXPECT_EQ(matchedR, 3);
}

TEST(HopcroftKarp, MatchConsistency) {
  // After matching, match_l[l]=r iff match_r[r]=l for all l,r.
  HopcroftKarp hk(5, 5, /*max_k=*/3);
  hk.addEdge(0, 0); hk.addEdge(0, 2);
  hk.addEdge(1, 1); hk.addEdge(1, 3);
  hk.addEdge(2, 0); hk.addEdge(2, 2); hk.addEdge(2, 4);
  hk.addEdge(3, 1); hk.addEdge(3, 3);
  hk.addEdge(4, 4);
  hk.maxMatching();
  for (int l = 0; l < 5; ++l) {
    if (hk.match_l[l] != -1) { EXPECT_EQ(hk.match_r[hk.match_l[l]], l) << "l=" << l; }
  }
  for (int r = 0; r < 5; ++r) {
    if (hk.match_r[r] != -1) { EXPECT_EQ(hk.match_l[hk.match_r[r]], r) << "r=" << r; }
  }
}

TEST(HopcroftKarp, LargerBipartite_3x3Grid) {
  // Full 3×3 grid bipartite: L=5 (even i+j), R=4 (odd). Max matching = 4.
  // L: 0=(1,1), 1=(1,3), 2=(2,2), 3=(3,1), 4=(3,3)
  // R: 0=(1,2), 1=(2,1), 2=(2,3), 3=(3,2)
  HopcroftKarp hk(5, 4, /*max_k=*/4);
  hk.addEdge(0, 0); hk.addEdge(0, 1);
  hk.addEdge(1, 0); hk.addEdge(1, 2);
  hk.addEdge(2, 0); hk.addEdge(2, 1); hk.addEdge(2, 2); hk.addEdge(2, 3);
  hk.addEdge(3, 1); hk.addEdge(3, 3);
  hk.addEdge(4, 2); hk.addEdge(4, 3);
  EXPECT_EQ(hk.maxMatching(), 4);
  for (int r = 0; r < 4; ++r) { EXPECT_NE(hk.match_r[r], -1) << "r=" << r; }
  int freeL = 0;
  for (int l = 0; l < 5; ++l) {
    if (hk.match_l[l] == -1) { ++freeL; }
  }
  EXPECT_EQ(freeL, 1);
}

TEST(HopcroftKarp, GridDerived_KnownMatchingSize) {
  // p=3, q=5, s=97: 3^i+5^j for i,j in [1,7] never reaches 97 (max=81+40=121,
  // 121%97=24), so no squares are removed and we get a clean full grid.

  // Full 4×4 grid: L=8, R=8, perfect matching exists -> max matching = 8.
  {
    auto hk = buildGridHK(4, 4, 3, 5, 97);
    EXPECT_EQ(hk.nL, 8);
    EXPECT_EQ(hk.nR, 8);
    EXPECT_EQ(hk.maxMatching(), 8);
    for (int l = 0; l < hk.nL; ++l) { EXPECT_NE(hk.match_l[l], -1); }
    for (int r = 0; r < hk.nR; ++r) { EXPECT_NE(hk.match_r[r], -1); }
  }
  // 1×N strip (path graph): L=⌈N/2⌉, R=⌊N/2⌋. Max matching = ⌊N/2⌋.
  for (int N : {3, 4, 5, 6, 7}) {
    auto hk = buildGridHK(1, N, 3, 5, 97);
    EXPECT_EQ(hk.nL + hk.nR, N) << "N=" << N; // no removals
    EXPECT_EQ(hk.maxMatching(), N / 2) << "N=" << N;
  }
}

TEST(HopcroftKarp, GridDerived_p419_q211) {
  // Use actual puzzle parameters for small grids; verify consistency.
  for (int s : {3, 5, 7, 11, 13}) {
    auto hk = buildGridHK(10, 10, 419, 211, s);
    int m = hk.maxMatching();
    // Matching must not exceed min(nL, nR)
    EXPECT_LE(m, std::min(hk.nL, hk.nR));
    // Consistency check
    for (int l = 0; l < hk.nL; ++l) {
      if (hk.match_l[l] != -1) { EXPECT_EQ(hk.match_r[hk.match_l[l]], l); }
    }
    for (int r = 0; r < hk.nR; ++r) {
      if (hk.match_r[r] != -1) { EXPECT_EQ(hk.match_l[hk.match_r[r]], r); }
    }
  }
}

// ===========================================================================
// Benchmarks for HopcroftKarp on large grids
//
// Graphs are derived from 1557×1557 boards with p=419, q=211 and various
// primes s.  Different primes give different removal densities (~1/s of
// squares removed), exercising both sparse and dense graph regimes.
//
// PauseTiming() / ResumeTiming() isolate matching from graph construction.
// ===========================================================================

static void BM_HopcroftKarp_1557x1557(benchmark::State& state, int s) {
  for (auto _ : state) {
    state.PauseTiming();
    auto hk = buildGridHK(1557, 1557, 419, 211, s);
    state.ResumeTiming();
    int m = hk.maxMatching();
    benchmark::DoNotOptimize(m);
  }
}

// s=3: ~1/3 squares removed, moderate-density graph
BENCHMARK_CAPTURE(BM_HopcroftKarp_1557x1557, s3,  3)->Unit(benchmark::kMillisecond);
// s=5: ~1/5 squares removed
BENCHMARK_CAPTURE(BM_HopcroftKarp_1557x1557, s5,  5)->Unit(benchmark::kMillisecond);
// s=7: ~1/7 squares removed
BENCHMARK_CAPTURE(BM_HopcroftKarp_1557x1557, s7,  7)->Unit(benchmark::kMillisecond);
// s=97: ~1/97 squares removed, near-full grid (dense, close to perfect matching)
BENCHMARK_CAPTURE(BM_HopcroftKarp_1557x1557, s97, 97)->Unit(benchmark::kMillisecond);

// Benchmark graph construction separately (so matching cost can be inferred)
static void BM_BuildGridHK_1557x1557(benchmark::State& state, int s) {
  for (auto _ : state) {
    auto hk = buildGridHK(1557, 1557, 419, 211, s);
    benchmark::DoNotOptimize(hk.adj.data());
  }
}
BENCHMARK_CAPTURE(BM_BuildGridHK_1557x1557, s3,  3)->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_BuildGridHK_1557x1557, s97, 97)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// Board classification for a single prime s
//
// Grid is N x M (1-indexed). Bipartite split:
//   L: squares where (i+j) is even   (index in [0, nL))
//   R: squares where (i+j) is odd    (index in [0, nR))
//
// Returns {countA, countB}.
// ---------------------------------------------------------------------------
struct Counts { long long A = 0, B = 0; };

Counts classifyBoard(int N, int M, int64_t p, int64_t q, int s) {
  // Precompute p^i mod s for i in [1,N] and q^j mod s for j in [1,M]
  std::vector<int> powP(N + 1), powQ(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * p % s; powP[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= M; ++j) { acc = acc * q % s; powQ[j] = (int)acc; }
  }

  int total = N * M;
  std::vector<int> lIdx(total, -1), rIdx(total, -1);
  int nL = 0, nR = 0;
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      if ((powP[i] + powQ[j]) % s == 0) { continue; } // removed
      int flat = (i - 1) * M + (j - 1);
      if ((i + j) % 2 == 0) {
        lIdx[flat] = nL++;
      } else {
        rIdx[flat] = nR++;
      }
    }
  }

  if (nL == 0 || nR == 0) {
    // No edges possible; all remaining nodes are "A" (isolated = free)
    return {nL + nR, 0};
  }

  // Build bipartite graph (L->R edges only); grid nodes have at most 4 neighbors
  HopcroftKarp hk(nL, nR, /*max_k=*/4);
  auto flat = [&](int i, int j) { return (i - 1) * M + (j - 1); };
  const int di[] = {0, 0, 1, -1};
  const int dj[] = {1, -1, 0, 0};
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      int f = flat(i, j);
      if (lIdx[f] == -1) { continue; } // removed or R-node
      for (int d = 0; d < 4; ++d) {
        int ni = i + di[d], nj = j + dj[d];
        if (ni < 1 || ni > N || nj < 1 || nj > M) { continue; }
        int nf = flat(ni, nj);
        if (rIdx[nf] == -1) { continue; } // removed or L-node
        hk.addEdge(lIdx[f], rIdx[nf]);
      }
    }
  }

  hk.maxMatching();

  // DM decomposition
  // Build directed graph D:
  //   matched edge (l,r):   arc r -> l  (fromR)
  //   unmatched edge (l,r): arc l -> r  (fromL)
  // D^T:
  //   unmatched edge (l,r): arc r -> l  (fromR_T)

  std::vector<std::vector<int>> fromL(nL), fromR(nR);
  std::vector<std::vector<int>> fromR_T(nR);

  for (int l = 0; l < nL; ++l) {
    for (int r : hk.adj_of(l)) {
      if (hk.match_l[l] == r) {
        fromR[r].push_back(l);
      } else {
        fromL[l].push_back(r);
        fromR_T[r].push_back(l);
      }
    }
  }

  // BFS 1: from free L-nodes in D, collect reachable L-nodes -> "A"
  std::vector<bool> notEveryL(nL, false);
  {
    std::queue<int> q;
    std::vector<bool> visitedL(nL, false), visitedR(nR, false);
    for (int l = 0; l < nL; ++l) {
      if (hk.match_l[l] == -1) {
        visitedL[l] = true;
        notEveryL[l] = true;
        q.push(l); // encode L-nodes as l, R-nodes as nL+r
      }
    }
    while (!q.empty()) {
      int node = q.front(); q.pop();
      if (node < nL) {
        // L-node: follow unmatched edges l->r
        for (int r : fromL[node]) {
          if (!visitedR[r]) {
            visitedR[r] = true;
            q.push(nL + r);
          }
        }
      } else {
        // R-node: follow matched edge r->l
        int r = node - nL;
        for (int l : fromR[r]) {
          if (!visitedL[l]) {
            visitedL[l] = true;
            notEveryL[l] = true;
            q.push(l);
          }
        }
      }
    }
  }

  // BFS 2: from free R-nodes in D^T, collect reachable R-nodes -> "A"
  std::vector<bool> notEveryR(nR, false);
  {
    std::vector<std::vector<int>> fromL_T(nL);
    for (int r = 0; r < nR; ++r) {
      for (int l : fromR[r]) {
        fromL_T[l].push_back(r);
      }
    }

    std::queue<int> q;
    std::vector<bool> visitedL(nL, false), visitedR(nR, false);
    for (int r = 0; r < nR; ++r) {
      if (hk.match_r[r] == -1) {
        visitedR[r] = true;
        notEveryR[r] = true;
        q.push(nL + r);
      }
    }
    while (!q.empty()) {
      int node = q.front(); q.pop();
      if (node >= nL) {
        int r = node - nL;
        for (int l : fromR_T[r]) {
          if (!visitedL[l]) {
            visitedL[l] = true;
            q.push(l);
          }
        }
      } else {
        int l = node;
        for (int r : fromL_T[l]) {
          if (!visitedR[r]) {
            visitedR[r] = true;
            notEveryR[r] = true;
            q.push(nL + r);
          }
        }
      }
    }
  }

  Counts cnt;
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      int f = flat(i, j);
      if ((i + j) % 2 == 0) {
        if (lIdx[f] == -1) { continue; }
        (notEveryL[lIdx[f]] ? cnt.A : cnt.B)++;
      } else {
        if (rIdx[f] == -1) { continue; }
        (notEveryR[rIdx[f]] ? cnt.A : cnt.B)++;
      }
    }
  }
  return cnt;
}

// ---------------------------------------------------------------------------
// Solve: sum over all primes s in S
// ---------------------------------------------------------------------------
Counts solve(int N, int M, int64_t p, int64_t q, int primeLimit) {
  auto primes = primesBelow(primeLimit);
  long long totalA = 0, totalB = 0;
  // schedule(dynamic) because runtime varies across primes (fewer removals =>
  // larger graph => more work), so static chunk assignment would leave some
  // threads idle while others finish the heavy large-prime cases.
  #pragma omp parallel for reduction(+:totalA,totalB) schedule(dynamic)
  for (int i = 0; i < (int)primes.size(); ++i) {
    const int s = primes[i];
    auto c = classifyBoard(N, M, p, q, s);
    totalA += c.A;
    totalB += c.B;
    LOG(INFO) << "s=" << s << " A=" << c.A << " B=" << c.B;
  }
  return {totalA, totalB};
}

// ===========================================================================
// Unit Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: brute-force vertex geography by game-tree search (for small boards)
// Returns true if the first player (Bob) wins starting at (si,sj).
// ---------------------------------------------------------------------------
bool firstPlayerWinsVG(int N, int M,
                       const std::vector<std::vector<bool>>& removed,
                       int si, int sj) {
  assert(N * M <= 25);
  int sz = N * M;
  auto idx = [&](int i, int j) { return i * M + j; };
  uint32_t initMask = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (removed[i][j]) { initMask |= (1u << idx(i, j)); }
    }
  }

  std::unordered_map<uint64_t, bool> memo;
  std::function<bool(int, uint32_t)> wins = [&](int pos, uint32_t mask) -> bool {
    uint64_t key = ((uint64_t)pos << 32) | mask;
    auto it = memo.find(key);
    if (it != memo.end()) { return it->second; }

    int ci = pos / M, cj = pos % M;
    const int di[] = {0, 0, 1, -1};
    const int dj[] = {1, -1, 0, 0};
    bool canMove = false;
    for (int d = 0; d < 4; ++d) {
      int ni = ci + di[d], nj = cj + dj[d];
      if (ni < 0 || ni >= N || nj < 0 || nj >= M) { continue; }
      int npos = idx(ni, nj);
      if (mask & (1u << npos)) { continue; } // removed
      canMove = true;
      uint32_t newMask = mask | (1u << pos); // remove current square
      if (!wins(npos, newMask)) { // opponent loses -> we win
        memo[key] = true;
        return true;
      }
    }
    if (!canMove) { memo[key] = false; return false; } // lose
    memo[key] = false;
    return false;
  };

  uint32_t mask = initMask;
  return wins(idx(si, sj), mask);
}

// Classify all squares of an N×M board with given removed mask via brute force.
Counts bruteForceBoard(int N, int M,
                       const std::vector<std::vector<bool>>& removed) {
  Counts cnt;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      if (removed[i][j]) { continue; }
      if (firstPlayerWinsVG(N, M, removed, i, j)) {
        cnt.B++;
      } else {
        cnt.A++;
      }
    }
  }
  return cnt;
}

// Build removed mask for classifyBoard (uses 1-indexed i,j internally)
std::vector<std::vector<bool>> buildRemoved(int N, int M, int64_t p, int64_t q,
                                            int s) {
  std::vector<bool> powP(N + 1), powQ(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * p % s; powP[i] = (acc == 0); }
    // recompute properly
  }
  std::vector<int> pp(N + 1), pq(M + 1);
  {
    int64_t acc = 1;
    for (int i = 1; i <= N; ++i) { acc = acc * (p % s) % s; pp[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= M; ++j) { acc = acc * (q % s) % s; pq[j] = (int)acc; }
  }
  std::vector<std::vector<bool>> removed(N, std::vector<bool>(M, false));
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= M; ++j) {
      removed[i - 1][j - 1] = ((pp[i] + pq[j]) % s == 0);
    }
  }
  return removed;
}

// ---------------------------------------------------------------------------
// Test: baseline 3x3 board (no removals) = alternating A/B
// ---------------------------------------------------------------------------
TEST(VertexGeography, BaselinePattern) {
  std::vector<std::vector<bool>> removed(3, std::vector<bool>(3, false));
  auto cnt = bruteForceBoard(3, 3, removed);
  EXPECT_EQ(cnt.A, 5);
  EXPECT_EQ(cnt.B, 4);
}

// ---------------------------------------------------------------------------
// Test: removing center of 3x3 gives all B
// ---------------------------------------------------------------------------
TEST(VertexGeography, Remove3x3Center) {
  std::vector<std::vector<bool>> removed(3, std::vector<bool>(3, false));
  removed[1][1] = true; // center
  auto cnt = bruteForceBoard(3, 3, removed);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 8);
}

// ---------------------------------------------------------------------------
// Test: isolated square is always A
// ---------------------------------------------------------------------------
TEST(VertexGeography, IsolatedSquare) {
  std::vector<std::vector<bool>> removed(1, std::vector<bool>(1, false));
  auto cnt = bruteForceBoard(1, 1, removed);
  EXPECT_EQ(cnt.A, 1);
  EXPECT_EQ(cnt.B, 0);
}

// ---------------------------------------------------------------------------
// Test: 1x2 board -> one L, one R, matched -> both B
// ---------------------------------------------------------------------------
TEST(VertexGeography, Board1x2) {
  std::vector<std::vector<bool>> removed(1, std::vector<bool>(2, false));
  auto cnt = bruteForceBoard(1, 2, removed);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 2);
}

// ---------------------------------------------------------------------------
// Test: 1x3 path -> A B A (center is B, ends are A)
// ---------------------------------------------------------------------------
TEST(VertexGeography, Board1x3) {
  std::vector<std::vector<bool>> removed(1, std::vector<bool>(3, false));
  auto cnt = bruteForceBoard(1, 3, removed);
  EXPECT_EQ(cnt.A, 2);
  EXPECT_EQ(cnt.B, 1);
}

// ---------------------------------------------------------------------------
// Test: classifyBoard matches brute force for small boards
// ---------------------------------------------------------------------------
void checkMatchesBruteForce(int N, int M, int64_t p, int64_t q, int s) {
  auto removed = buildRemoved(N, M, p, q, s);
  auto expected = bruteForceBoard(N, M, removed);
  auto got = classifyBoard(N, M, p, q, s);
  EXPECT_EQ(got.A, expected.A)
      << "N=" << N << " M=" << M << " p=" << p << " q=" << q << " s=" << s;
  EXPECT_EQ(got.B, expected.B)
      << "N=" << N << " M=" << M << " p=" << p << " q=" << q << " s=" << s;
}

TEST(ClassifyBoard, SmallBoards_p19_q2) {
  for (int s : {2, 3, 5, 7, 11}) {
    checkMatchesBruteForce(3, 3, 19, 2, s);
    checkMatchesBruteForce(2, 4, 19, 2, s);
    checkMatchesBruteForce(4, 2, 19, 2, s);
    checkMatchesBruteForce(3, 4, 19, 2, s);
  }
}

TEST(ClassifyBoard, SmallBoards_p7_q3) {
  for (int s : {2, 3, 5, 7, 11, 13}) {
    checkMatchesBruteForce(3, 3, 7, 3, s);
    checkMatchesBruteForce(2, 3, 7, 3, s);
    checkMatchesBruteForce(4, 3, 7, 3, s);
  }
}

TEST(ClassifyBoard, SmallBoards_p5_q11) {
  for (int s : {2, 3, 5, 7, 11}) {
    checkMatchesBruteForce(3, 3, 5, 11, s);
    checkMatchesBruteForce(2, 4, 5, 11, s);
  }
}

TEST(ClassifyBoard, AllSquaresRemoved) {
  auto cnt = classifyBoard(5, 5, 1, 1, 2);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 0);
}

TEST(ClassifyBoard, NothingRemoved_SmallBoards) {
  for (int N = 1; N <= 4; ++N) {
    for (int M = 1; M <= 4; ++M) {
      if (N * M > 16) { continue; }
      std::vector<std::vector<bool>> removed(N, std::vector<bool>(M, false));
      auto expected = bruteForceBoard(N, M, removed);
      bool anyRemoved = false;
      for (int i = 1; i <= N && !anyRemoved; ++i) {
        int64_t pi = 1;
        for (int k = 0; k < i; ++k) { pi = pi * 2 % 97; }
        for (int j = 1; j <= M && !anyRemoved; ++j) {
          int64_t qj = 1;
          for (int k = 0; k < j; ++k) { qj = qj * 3 % 97; }
          if ((pi + qj) % 97 == 0) { anyRemoved = true; }
        }
      }
      if (!anyRemoved) {
        auto got = classifyBoard(N, M, 2, 3, 97);
        EXPECT_EQ(got.A, expected.A) << "N=" << N << " M=" << M;
        EXPECT_EQ(got.B, expected.B) << "N=" << N << " M=" << M;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Test: the example from the puzzle statement
// N=M=3, p=19, q=2, S=[2,3,5,7,11] -> total A=16, B=19
// ---------------------------------------------------------------------------
TEST(Solve, PuzzleExample) {
  long long totalA = 0, totalB = 0;
  for (int s : {2, 3, 5, 7, 11}) {
    auto c = classifyBoard(3, 3, 19, 2, s);
    totalA += c.A;
    totalB += c.B;
  }
  EXPECT_EQ(totalA, 16);
  EXPECT_EQ(totalB, 19);
}

// ---------------------------------------------------------------------------
// Test: verify solve() helper accumulates correctly
// ---------------------------------------------------------------------------
TEST(Solve, SolveHelperMatchesManual) {
  auto manual = [](int N, int M, int64_t p, int64_t q,
                   std::vector<int> primes) -> Counts {
    Counts total;
    for (int s : primes) {
      auto c = classifyBoard(N, M, p, q, s);
      total.A += c.A;
      total.B += c.B;
    }
    return total;
  };

  auto s1 = solve(3, 3, 19, 2, 12); // primes < 12: {2,3,5,7,11}
  auto m1 = manual(3, 3, 19, 2, {2, 3, 5, 7, 11});
  EXPECT_EQ(s1.A, m1.A);
  EXPECT_EQ(s1.B, m1.B);

  auto s2 = solve(4, 4, 7, 3, 10); // primes < 10: {2,3,5,7}
  auto m2 = manual(4, 4, 7, 3, {2, 3, 5, 7});
  EXPECT_EQ(s2.A, m2.A);
  EXPECT_EQ(s2.B, m2.B);
}

TEST(ClassifyBoard, SpecificSquare_3x3_s5_p19_q2) {
  auto cnt = classifyBoard(3, 3, 19, 2, 5);
  EXPECT_EQ(cnt.A, 0);
  EXPECT_EQ(cnt.B, 8);
}

TEST(ClassifyBoard, SpecificSquare_3x3_s2_p19_q2) {
  auto cnt = classifyBoard(3, 3, 19, 2, 2);
  EXPECT_EQ(cnt.A, 5);
  EXPECT_EQ(cnt.B, 4);
}

TEST(ClassifyBoard, MediumBoards_BruteForceCheck) {
  struct Case { int N, M; int64_t p, q; int s; };
  std::vector<Case> cases = {
    {4, 4, 19, 2, 3},
    {4, 4, 19, 2, 5},
    {4, 4, 19, 2, 7},
    {4, 4,  7, 3, 5},
    {4, 4,  7, 3, 7},
    {3, 4, 19, 2, 5},
    {4, 3, 19, 2, 5},
    {4, 4, 13, 7, 3},
    {4, 4, 13, 7, 11},
  };
  for (auto& c : cases) {
    checkMatchesBruteForce(c.N, c.M, c.p, c.q, c.s);
  }
}

TEST(ClassifyBoard, RectangularBoards) {
  for (int s : {3, 5, 7}) {
    checkMatchesBruteForce(2, 3, 19, 2, s);
    checkMatchesBruteForce(3, 2, 19, 2, s);
    checkMatchesBruteForce(2, 4, 7, 5, s);
    checkMatchesBruteForce(4, 2, 7, 5, s);
  }
}

TEST(Sieve, PrimesBelow100) {
  auto primes = primesBelow(100);
  EXPECT_EQ(primes.size(), 25u);
  EXPECT_EQ(primes.front(), 2);
  EXPECT_EQ(primes.back(), 97);
}

TEST(Sieve, PrimesBelow12) {
  auto primes = primesBelow(12);
  EXPECT_EQ(primes, (std::vector<int>{2, 3, 5, 7, 11}));
}

TEST(Sanity, ABSumEqualsNonRemoved) {
  struct Case { int N, M; int64_t p, q; int s; };
  std::vector<Case> cases = {
    {3, 3, 19, 2, 2}, {3, 3, 19, 2, 5}, {4, 4, 7, 3, 3}, {5, 5, 11, 7, 5},
  };
  for (auto& c : cases) {
    std::vector<int> pp(c.N + 1), pq(c.M + 1);
    int64_t acc = 1;
    for (int i = 1; i <= c.N; ++i) { acc = acc * (c.p % c.s) % c.s; pp[i] = (int)acc; }
    acc = 1;
    for (int j = 1; j <= c.M; ++j) { acc = acc * (c.q % c.s) % c.s; pq[j] = (int)acc; }
    int nonRemoved = 0;
    for (int i = 1; i <= c.N; ++i) {
      for (int j = 1; j <= c.M; ++j) {
        if ((pp[i] + pq[j]) % c.s != 0) { ++nonRemoved; }
      }
    }
    auto cnt = classifyBoard(c.N, c.M, c.p, c.q, c.s);
    EXPECT_EQ(cnt.A + cnt.B, nonRemoved)
        << "N=" << c.N << " M=" << c.M << " s=" << c.s;
  }
}

TEST(ClassifyBoard, Board5x4_BruteForce) {
  for (int s : {3, 5, 7}) {
    checkMatchesBruteForce(4, 4, 19, 2, s);
  }
}

// ---------------------------------------------------------------------------
// Test: 5×5 board — largest size the brute-force game-tree search can handle
// (assert in firstPlayerWinsVG limits N*M <= 25).
//
// p=19, q=2, s=3:  19≡1 mod 3 and 2^j≡{2,1,2,1,2} mod 3 for j=1..5,
//   so (1 + 2^j) mod 3 = 0 for j odd.  All 15 squares with j in {1,3,5} are
//   removed; only columns j=2,4 survive (10 squares, fast for brute-force).
//   Those two columns are independent 5×1 paths; no cross-column edges remain.
//
// p=19, q=2, s=5:  19≡4 mod 5, 2^j mod 5 cycles {2,4,3,1,2,...}.
//   Removed squares: (4+2^j)%5==0 => 2^j≡1 mod 5 => j≡0 mod 4 => j=4 only.
//   5 squares removed (all i, j=4); 20 survive — still feasible.
// ---------------------------------------------------------------------------
TEST(ClassifyBoard, Board5x5_BruteForce) {
  checkMatchesBruteForce(5, 5, 19, 2, 3);
  checkMatchesBruteForce(5, 5, 19, 2, 5);
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char** argv) {
  // Initialize in this order so each library consumes only its own flags.
  benchmark::Initialize(&argc, argv);   // strips --benchmark_* flags
  testing::InitGoogleTest(&argc, argv); // strips --gtest_* flags
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  auto res = RUN_ALL_TESTS();
  if (res == 0 && argc <= 1) {
    benchmark::RunSpecifiedBenchmarks();
  }

  if (argc <= 1 || std::string(argv[1]) != "solve") {
    return res;
  }

  if (argc < 3 || std::string(argv[2]) == "main") {
    LOG(INFO) << "Solving main puzzle: N=M=157, p=419, q=211, primes < 100";
    auto main_ans = solve(157, 157, 419, 211, 100);
    std::cout << "Main: " << main_ans.A << " " << main_ans.B << std::endl;
  }
  if (argc < 3 || std::string(argv[2]) == "bonus") {
    LOG(INFO) << "Solving bonus: N=M=1557, p=419, q=211, primes < 500";
    auto bonus_ans = solve(1557, 1557, 419, 211, 500);
    std::cout << "Bonus: " << bonus_ans.A << " " << bonus_ans.B << std::endl;
  }
  return res;
}
