/*
  Source code for the puzzle
  https://research.ibm.com/blog/ponder-this-july-2026

  Return of the Superheroes: n superheroes 1..n are paired with n
  supervillains 1..n by a perfect matching. For a pair (a, b) define
      T_{a,b}(x) = x^2 + a*x + b  (mod p),   x_0 = 0,  x_{k+1} = T(x_k).
  The sequence eventually repeats; f(a, b) is the index of the first element
  equal to an earlier element, i.e. the number of distinct elements in the
  orbit of 0 (its "rho length").

  The hero-villain value is the maximum, over all perfect matchings, of the
  minimum f(a, b) in the matching — a bottleneck assignment problem.

  Main goal : the hero-villain value for n = 611, p = 14411.
  Bonus (*) : the n with 1 < n < 1000 maximizing the hero-villain value
              for p = 17377.

  Approach:
    1. f(a, b) in O(rho) time with a timestamped last-seen table; all n^2
       values computed in parallel (rho ~ sqrt(p) on average).
    2. Threshold-descending incremental matching: insert edges in decreasing
       f order while maintaining a maximum matching. Inserting (u, v) grows
       the matching iff there is an augmenting path through (u, v), found by
       two independent alternating-path searches from the endpoints of the
       new edge. The answer is the f value of the edge whose insertion first
       makes the matching perfect.
    3. An n!-permutation bruteForce() cross-checks the solver for small n
       in tests.

  C++20 features used: std::jthread (auto-joining threads), ranges/views,
  std::span, std::string_view over argv, [[nodiscard]]. fmt::print stands
  in for C++23 std::print (GCC 11 ships neither std::format nor std::print).
*/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <numeric>
#include <omp.h>
#include <ranges>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/core.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// f(a, b): steps of x -> x^2 + a*x + b (mod p) from x = 0 until the first
// value that already appeared. lastSeen must have size p; entries equal to
// `stamp` mark values visited in this call, so the caller passes a fresh
// stamp each call and never needs to clear the table.
// ---------------------------------------------------------------------------
[[nodiscard]] int rhoLength(int64_t a, int64_t b, int p,
                            std::vector<int>& lastSeen, int stamp) {
  int64_t x = 0;
  int steps = 0;
  while (lastSeen[x] != stamp) {
    lastSeen[x] = stamp;
    x = (x * x + a * x + b) % p;
    ++steps;
  }
  return steps;
}

// ---------------------------------------------------------------------------
// Full f matrix for heroes/villains 1..n, stored row-major with stride n:
// f[(a-1)*n + (b-1)] = f(a, b). Rows are partitioned across threads.
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<int> fMatrix(int n, int p) {
  std::vector<int> f(static_cast<size_t>(n) * n);
  const int numThreads = std::max(1, std::min(n, omp_get_max_threads()));
  {
    // std::jthread joins on destruction, so leaving this scope is the
    // synchronization point.
    std::vector<std::jthread> workers;
    for (int tid = 0; tid < numThreads; ++tid) {
      workers.emplace_back([&f, n, p, numThreads, tid] {
        std::vector<int> lastSeen(p, -1);
        int stamp = 0;
        for (int a = 1 + tid; a <= n; a += numThreads) {
          for (const int b : std::views::iota(1, n + 1)) {
            f[static_cast<size_t>(a - 1) * n + (b - 1)] =
                rhoLength(a, b, p, lastSeen, stamp++);
          }
        }
      });
    }
  }
  return f;
}

// ---------------------------------------------------------------------------
// Incremental maximum bipartite matching between n heroes and n villains: a
// maximum matching is maintained after every single-edge insertion, at
// O(V + E) worst-case cost per insertion.
//
// Inserting (u, v) can grow the matching by at most 1, and only via an
// augmenting path that uses (u, v) itself: an augmenting path avoiding
// (u, v) would contradict the maximality of the current matching. Such a
// path exists iff
//   - some free hero reaches u along an alternating path, and
//   - v reaches some free villain along an alternating path,
// which two DFS calls decide directly. No BFS layering (as in
// Hopcroft-Karp) is needed: shortest augmenting paths only matter for
// Hopcroft-Karp's O(E*sqrt(V)) phase bound. By Berge's theorem a matching
// is maximum iff no augmenting path exists at all, so augmenting along an
// arbitrary augmenting path after each insertion keeps the matching
// maximum.
//
// The two searches can run independently: if the current matching is
// maximum, any left path and right path are automatically vertex-disjoint.
// Otherwise, shortcutting at the first shared vertex would splice together
// an augmenting path that avoids (u, v) entirely — contradiction. Each
// search flips the matched edges along its path on success, preserving the
// matching size and leaving its endpoint free, so (u, v) can then be
// matched directly.
//
// Heroes and villains share one id space: hero u is node 2*u and villain v
// is node 2*v + 1, which lets a single search() serve both sides and a
// single match_ array store both directions. A node of parity P has a
// partner of parity 1-P whose neighbors have parity P again, so a search
// never leaves the side it started on; the searches from the two endpoints
// of a new edge touch disjoint ids and share visited_.
// ---------------------------------------------------------------------------
class IncrementalMatcher {
 public:
  explicit IncrementalMatcher(int n)
      : adj_(2 * n), match_(2 * n, -1), visited_(2 * n, -1) {}

  [[nodiscard]] int matchedCount() const { return matched_; }

  // Inserts the edge (u, v) and returns true iff the matching grew.
  [[nodiscard]] bool addEdge(int u, int v) {
    const int hero = 2 * u;
    const int villain = 2 * v + 1;
    adj_[hero].push_back(villain);
    adj_[villain].push_back(hero);
    ++stamp_;
    // A failed second search needs no rollback: the flips done by a
    // successful first search only shuffle partners along an alternating
    // path, so the matching stays valid, maximum, and of the same size.
    if (!search(hero) || !search(villain)) {
      return false;
    }
    match_[hero] = villain;
    match_[villain] = hero;
    ++matched_;
    return true;
  }

 private:
  // Frees `node` by shifting matched partners along an alternating path
  // ending at a free node on the same side. Returns false if no such path
  // exists.
  bool search(int node) {
    visited_[node] = stamp_;
    if (match_[node] == -1) {
      return true;
    }
    const int partner = match_[node];
    for (const int next : adj_[partner]) {
      // (next, partner) is an unmatched edge for every next != node, and
      // node is already visited.
      if (visited_[next] != stamp_ && search(next)) {
        match_[next] = partner;
        match_[partner] = next;
        match_[node] = -1;
        return true;
      }
    }
    return false;
  }

  std::vector<std::vector<int>> adj_;
  std::vector<int> match_, visited_;
  int stamp_ = 0;
  int matched_ = 0;
};

// ---------------------------------------------------------------------------
// Bottleneck assignment: max over perfect matchings of the min f in the
// matching. f is row-major with the given stride; the top-left n x n
// submatrix is used (heroes/villains 1..n). Threshold-descending: insert
// edges in decreasing f order; the answer is the f value of the edge whose
// insertion first makes the matching perfect. In practice only the top
// O(n log n) edges are inserted before that happens.
// ---------------------------------------------------------------------------
[[nodiscard]] int heroVillainValue(const std::vector<int>& f, int stride,
                                   int n) {
  // A bounds-carrying view (std::span) of row a's first n entries.
  const auto row = [&f, stride, n](int a) {
    return std::span(f).subspan(static_cast<size_t>(a) * stride, n);
  };
  int maxF = 0;
  for (const int a : std::views::iota(0, n)) {
    maxF = std::max(maxF, std::ranges::max(row(a)));
  }
  // Bucket the edges by f value (counting sort); f(a, b) >= 1 always.
  std::vector<std::vector<std::pair<int, int>>> edgesByValue(maxF + 1);
  for (const int a : std::views::iota(0, n)) {
    for (const int b : std::views::iota(0, n)) {
      edgesByValue[row(a)[b]].emplace_back(a, b);
    }
  }
  IncrementalMatcher matcher(n);
  for (int value = maxF; value > 0; --value) {
    for (const auto& [a, b] : edgesByValue[value]) {
      if (matcher.addEdge(a, b) && matcher.matchedCount() == n) {
        return value;
      }
    }
  }
  LOG(FATAL) << "K_{n,n} always has a perfect matching";
  return 0;
}

[[nodiscard]] int bruteForce(const std::vector<int>& f, int stride, int n) {
  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  int best = 0;
  do {
    // The matching's min value as a lazy view pipeline.
    const int minF = std::ranges::min(
        std::views::iota(0, n) | std::views::transform([&](int a) {
          return f[static_cast<size_t>(a) * stride + perm[a]];
        }));
    best = std::max(best, minF);
  } while (std::ranges::next_permutation(perm).found);
  return best;
}

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------
void solveMain() {
  constexpr int kN = 611;
  constexpr int kP = 14411;
  const auto f = fMatrix(kN, kP);
  fmt::print("==> Hero-villain value for n = {}, p = {}: {}\n", kN, kP,
             heroVillainValue(f, kN, kN));
}

void solveBonus() {
  constexpr int kMaxN = 999;  // 1 < n < 1000
  constexpr int kP = 17377;
  const auto f = fMatrix(kMaxN, kP);
  std::vector<int> value(kMaxN + 1, 0);
  std::atomic<int> nextN{2};
  {
    // Worker jthreads pull n values dynamically; all join at scope exit.
    const int numThreads = std::max(1, omp_get_max_threads());
    std::vector<std::jthread> workers;
    for (int tid = 0; tid < numThreads; ++tid) {
      workers.emplace_back([&f, &value, &nextN] {
        for (int n = nextN.fetch_add(1); n <= kMaxN; n = nextN.fetch_add(1)) {
          value[n] = heroVillainValue(f, kMaxN, n);
        }
      });
    }
  }
  // value[0] and value[1] stay 0 (< any real value), so max_element over
  // the whole vector returns the smallest optimal n.
  const auto bestIt = std::ranges::max_element(value);
  const auto bestN = bestIt - value.begin();
  fmt::print("==> Bonus: optimal n for p = {} is {} with hero-villain value {}\n",
             kP, bestN, *bestIt);
  for (const int n : std::views::iota(2, kMaxN + 1)) {
    if (n != bestN && value[n] == *bestIt) {
      fmt::print("    (tie: n = {})\n", n);
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();
  if (res != 0) {
    return res;
  }
  bool doSolve = false;
  bool doBonus = false;
  for (const std::string_view arg : std::span(argv + 1, argc - 1)) {
    doSolve |= arg == "solve";
    doBonus |= arg == "bonus";
  }
  if (doSolve) {
    solveMain();
  }
  if (doBonus) {
    solveBonus();
  }
  return res;
}

// ---------------------------------------------------------------------------
// Tests. gtest registers TEST blocks at static-initialization time, so
// defining them after main() changes nothing about what RUN_ALL_TESTS()
// executes.
// ---------------------------------------------------------------------------
TEST(RhoLength, PuzzleExampleValues) {
  // From the puzzle statement: for p = 101 the pairings
  // (1,3), (2,1), (3,4), (4,2), (5,5) yield 14, 18, 19, 22, 14.
  constexpr int kP = 101;
  std::vector<int> lastSeen(kP, -1);
  int stamp = 0;
  EXPECT_EQ(rhoLength(1, 3, kP, lastSeen, stamp++), 14);
  EXPECT_EQ(rhoLength(2, 1, kP, lastSeen, stamp++), 18);
  EXPECT_EQ(rhoLength(3, 4, kP, lastSeen, stamp++), 19);
  EXPECT_EQ(rhoLength(4, 2, kP, lastSeen, stamp++), 22);
  EXPECT_EQ(rhoLength(5, 5, kP, lastSeen, stamp++), 14);
}

TEST(IncrementalMatcher, Basic) {
  {
    // Perfect matching exists: 0-0, 1-1.
    IncrementalMatcher m(2);
    EXPECT_TRUE(m.addEdge(0, 0));
    EXPECT_TRUE(m.addEdge(1, 1));
    EXPECT_EQ(m.matchedCount(), 2);
  }
  {
    // Both heroes only connect to villain 0.
    IncrementalMatcher m(2);
    EXPECT_TRUE(m.addEdge(0, 0));
    EXPECT_FALSE(m.addEdge(1, 0));
    EXPECT_EQ(m.matchedCount(), 1);
  }
  {
    // Inserting (1, 0) augments along villain 1 - hero 0 - villain 0:
    // hero 0 hands villain 0 over to hero 1 and takes villain 1.
    IncrementalMatcher m(2);
    EXPECT_TRUE(m.addEdge(0, 0));
    EXPECT_FALSE(m.addEdge(0, 1));  // hero 0 is already matched
    EXPECT_TRUE(m.addEdge(1, 0));
    EXPECT_EQ(m.matchedCount(), 2);
  }
}

TEST(HeroVillain, HandCraftedMatrices) {
  {
    // Diagonal matching gives 5; the other matching gives 1.
    const std::vector<int> f = {5, 1, 1, 5};
    EXPECT_EQ(heroVillainValue(f, 2, 2), 5);
  }
  {
    // Anti-diagonal matching gives min(9, 9) = 9; diagonal gives min(9, 1).
    const std::vector<int> f = {9, 9, 9, 1};
    EXPECT_EQ(heroVillainValue(f, 2, 2), 9);
  }
}

TEST(HeroVillain, PuzzleExample) {
  constexpr int kN = 5;
  constexpr int kP = 101;
  const auto f = fMatrix(kN, kP);
  EXPECT_EQ(heroVillainValue(f, kN, kN), 14);
  EXPECT_EQ(bruteForce(f, kN, kN), 14);
}

TEST(HeroVillain, SolveMatchesBruteForceOnSmallInputs) {
  for (const int p : {11, 31, 101, 997}) {
    for (int n = 2; n <= 7; ++n) {
      const auto f = fMatrix(n, p);
      EXPECT_EQ(heroVillainValue(f, n, n), bruteForce(f, n, n))
          << "n=" << n << " p=" << p;
    }
  }
}
