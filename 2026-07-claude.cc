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
    3. Cross-checks in tests: a binary-search-over-thresholds solver with
       Hopcroft-Karp feasibility checks, and an n!-permutation bruteForce()
       for small n.
*/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// f(a, b): steps of x -> x^2 + a*x + b (mod p) from x = 0 until the first
// value that already appeared. lastSeen must have size p; entries equal to
// `stamp` mark values visited in this call, so the caller passes a fresh
// stamp each call and never needs to clear the table.
// ---------------------------------------------------------------------------
int rhoLength(int64_t a, int64_t b, int p, std::vector<int>& lastSeen,
              int stamp) {
  int64_t x = 0;
  int steps = 0;
  while (lastSeen[x] != stamp) {
    lastSeen[x] = stamp;
    x = (x * x + a * x + b) % p;
    ++steps;
  }
  return steps;
}

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

// ---------------------------------------------------------------------------
// Full f matrix for heroes/villains 1..n, stored row-major with stride n:
// f[(a-1)*n + (b-1)] = f(a, b). Rows are partitioned across threads.
// ---------------------------------------------------------------------------
std::vector<int> fMatrix(int n, int p) {
  std::vector<int> f(static_cast<size_t>(n) * n);
  const int numThreads = std::max(1, std::min(n, omp_get_max_threads()));
  std::vector<std::thread> threads;
  for (int tid = 0; tid < numThreads; ++tid) {
    threads.emplace_back([&f, n, p, numThreads, tid] {
      std::vector<int> lastSeen(p, -1);
      int stamp = 0;
      for (int a = 1 + tid; a <= n; a += numThreads) {
        for (int b = 1; b <= n; ++b) {
          f[static_cast<size_t>(a - 1) * n + (b - 1)] =
              rhoLength(a, b, p, lastSeen, stamp++);
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  return f;
}

// ---------------------------------------------------------------------------
// Hopcroft-Karp maximum bipartite matching.
// ---------------------------------------------------------------------------
class BipartiteMatcher {
 public:
  BipartiteMatcher(int numLeft, int numRight)
      : numLeft_(numLeft), numRight_(numRight), adj_(numLeft) {}

  void addEdge(int left, int right) { adj_[left].push_back(right); }

  int maxMatchingSize() {
    matchLeft_.assign(numLeft_, -1);
    matchRight_.assign(numRight_, -1);
    dist_.assign(numLeft_, 0);
    int matched = 0;
    while (bfs()) {
      for (int u = 0; u < numLeft_; ++u) {
        if (matchLeft_[u] == -1 && dfs(u)) {
          ++matched;
        }
      }
    }
    return matched;
  }

 private:
  static constexpr int kInf = 1 << 29;

  bool bfs() {
    std::queue<int> queue;
    for (int u = 0; u < numLeft_; ++u) {
      if (matchLeft_[u] == -1) {
        dist_[u] = 0;
        queue.push(u);
      } else {
        dist_[u] = kInf;
      }
    }
    bool foundAugmentingPath = false;
    while (!queue.empty()) {
      const int u = queue.front();
      queue.pop();
      for (const int v : adj_[u]) {
        const int w = matchRight_[v];
        if (w == -1) {
          foundAugmentingPath = true;
        } else if (dist_[w] == kInf) {
          dist_[w] = dist_[u] + 1;
          queue.push(w);
        }
      }
    }
    return foundAugmentingPath;
  }

  bool dfs(int u) {
    for (const int v : adj_[u]) {
      const int w = matchRight_[v];
      if (w == -1 || (dist_[w] == dist_[u] + 1 && dfs(w))) {
        matchLeft_[u] = v;
        matchRight_[v] = u;
        return true;
      }
    }
    dist_[u] = kInf;
    return false;
  }

  int numLeft_, numRight_;
  std::vector<std::vector<int>> adj_;
  std::vector<int> matchLeft_, matchRight_, dist_;
};

TEST(BipartiteMatcher, Basic) {
  {
    // Perfect matching exists: 0-0, 1-1.
    BipartiteMatcher m(2, 2);
    m.addEdge(0, 0);
    m.addEdge(1, 1);
    EXPECT_EQ(m.maxMatchingSize(), 2);
  }
  {
    // Both left vertices only connect to right vertex 0.
    BipartiteMatcher m(2, 2);
    m.addEdge(0, 0);
    m.addEdge(1, 0);
    EXPECT_EQ(m.maxMatchingSize(), 1);
  }
  {
    // Augmenting path needed: 0 must give up vertex 0 to 1.
    BipartiteMatcher m(2, 2);
    m.addEdge(0, 0);
    m.addEdge(0, 1);
    m.addEdge(1, 0);
    EXPECT_EQ(m.maxMatchingSize(), 2);
  }
}

// ---------------------------------------------------------------------------
// Bottleneck assignment: max over perfect matchings of the min f in the
// matching. f is row-major with the given stride; the top-left n x n
// submatrix is used (heroes/villains 1..n).
// ---------------------------------------------------------------------------
bool perfectMatchingExists(const std::vector<int>& f, int stride, int n,
                           int minValue) {
  BipartiteMatcher matcher(n, n);
  for (int a = 0; a < n; ++a) {
    const int* row = f.data() + static_cast<size_t>(a) * stride;
    for (int b = 0; b < n; ++b) {
      if (row[b] >= minValue) {
        matcher.addEdge(a, b);
      }
    }
  }
  return matcher.maxMatchingSize() == n;
}

int heroVillainValue(const std::vector<int>& f, int stride, int n) {
  std::vector<int> vals;
  vals.reserve(static_cast<size_t>(n) * n);
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      vals.push_back(f[static_cast<size_t>(a) * stride + b]);
    }
  }
  std::sort(vals.begin(), vals.end());
  vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
  // The threshold vals[0] admits all n^2 edges, so it is always feasible.
  // Find the largest feasible threshold.
  size_t lo = 0;
  size_t hi = vals.size() - 1;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo + 1) / 2;
    if (perfectMatchingExists(f, stride, n, vals[mid])) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return vals[lo];
}

// ---------------------------------------------------------------------------
// Incremental bottleneck matcher: edges are inserted in decreasing f order
// and a maximum matching is maintained after every insertion. Inserting
// (u, v) can grow the matching by at most 1, and only via an augmenting path
// that uses (u, v) itself: an augmenting path avoiding (u, v) would
// contradict the maximality of the current matching. Such a path exists iff
//   - some free hero reaches u along an alternating path (leftSearch), and
//   - v reaches some free villain along an alternating path (rightSearch).
// The two searches can run independently: if the current matching is
// maximum, any left path and right path are automatically vertex-disjoint.
// Otherwise, shortcutting at the first shared vertex would splice together
// an augmenting path that avoids (u, v) entirely — contradiction. Each
// search flips the matched edges along its path on success, preserving the
// matching size and leaving its endpoint free, so (u, v) can then be
// matched directly.
//
// An insertion costs O(V + E) worst case, but the solver stops as soon as
// the matching becomes perfect, which happens after inserting only the top
// O(n log n) edges in practice.
// ---------------------------------------------------------------------------
class IncrementalMatcher {
 public:
  explicit IncrementalMatcher(int n)
      : adjLeft_(n),
        adjRight_(n),
        matchLeft_(n, -1),
        matchRight_(n, -1),
        visitedLeft_(n, -1),
        visitedRight_(n, -1) {}

  int matchedCount() const { return matched_; }

  // Inserts the edge (u, v) and returns true iff the matching grew.
  bool addEdge(int u, int v) {
    adjLeft_[u].push_back(v);
    adjRight_[v].push_back(u);
    ++stamp_;
    // A failed second search needs no rollback: the flips done by a
    // successful first search only shuffle partners along an alternating
    // path, so the matching stays valid, maximum, and of the same size.
    if (!leftSearch(u) || !rightSearch(v)) {
      return false;
    }
    matchLeft_[u] = v;
    matchRight_[v] = u;
    ++matched_;
    return true;
  }

 private:
  // Frees hero `l` by shifting matched partners along an alternating path
  // ending at a free hero. Returns false if no such path exists.
  bool leftSearch(int l) {
    visitedLeft_[l] = stamp_;
    if (matchLeft_[l] == -1) {
      return true;
    }
    const int r = matchLeft_[l];
    for (const int l2 : adjRight_[r]) {
      // (l2, r) is an unmatched edge for every l2 != l, and l is visited.
      if (visitedLeft_[l2] != stamp_ && leftSearch(l2)) {
        matchLeft_[l2] = r;
        matchRight_[r] = l2;
        matchLeft_[l] = -1;
        return true;
      }
    }
    return false;
  }

  // Frees villain `r` by shifting matched partners along an alternating
  // path ending at a free villain. Returns false if no such path exists.
  bool rightSearch(int r) {
    visitedRight_[r] = stamp_;
    if (matchRight_[r] == -1) {
      return true;
    }
    const int l = matchRight_[r];
    for (const int r2 : adjLeft_[l]) {
      if (visitedRight_[r2] != stamp_ && rightSearch(r2)) {
        matchRight_[r2] = l;
        matchLeft_[l] = r2;
        matchRight_[r] = -1;
        return true;
      }
    }
    return false;
  }

  std::vector<std::vector<int>> adjLeft_, adjRight_;
  std::vector<int> matchLeft_, matchRight_;
  std::vector<int> visitedLeft_, visitedRight_;
  int stamp_ = 0;
  int matched_ = 0;
};

int heroVillainValueIncremental(const std::vector<int>& f, int stride,
                                int n) {
  int maxF = 0;
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      maxF = std::max(maxF, f[static_cast<size_t>(a) * stride + b]);
    }
  }
  // Bucket the edges by f value (counting sort); f(a, b) >= 1 always.
  std::vector<std::vector<std::pair<int, int>>> edgesByValue(maxF + 1);
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      edgesByValue[f[static_cast<size_t>(a) * stride + b]].emplace_back(a, b);
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

int bruteForce(const std::vector<int>& f, int stride, int n) {
  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  int best = 0;
  do {
    int minF = std::numeric_limits<int>::max();
    for (int a = 0; a < n; ++a) {
      minF = std::min(minF, f[static_cast<size_t>(a) * stride + perm[a]]);
    }
    best = std::max(best, minF);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return best;
}

TEST(HeroVillain, HandCraftedMatrices) {
  {
    // Diagonal matching gives 5; the other matching gives 1.
    const std::vector<int> f = {5, 1, 1, 5};
    EXPECT_EQ(heroVillainValue(f, 2, 2), 5);
    EXPECT_EQ(heroVillainValueIncremental(f, 2, 2), 5);
  }
  {
    // Anti-diagonal matching gives min(9, 9) = 9; diagonal gives min(9, 1).
    const std::vector<int> f = {9, 9, 9, 1};
    EXPECT_EQ(heroVillainValue(f, 2, 2), 9);
    EXPECT_EQ(heroVillainValueIncremental(f, 2, 2), 9);
  }
}

TEST(HeroVillain, PuzzleExample) {
  constexpr int kN = 5;
  constexpr int kP = 101;
  const auto f = fMatrix(kN, kP);
  EXPECT_EQ(heroVillainValue(f, kN, kN), 14);
  EXPECT_EQ(heroVillainValueIncremental(f, kN, kN), 14);
  EXPECT_EQ(bruteForce(f, kN, kN), 14);
}

TEST(HeroVillain, SolveMatchesBruteForceOnSmallInputs) {
  for (const int p : {11, 31, 101, 997}) {
    for (int n = 2; n <= 7; ++n) {
      const auto f = fMatrix(n, p);
      const int expected = bruteForce(f, n, n);
      EXPECT_EQ(heroVillainValue(f, n, n), expected) << "n=" << n << " p=" << p;
      EXPECT_EQ(heroVillainValueIncremental(f, n, n), expected)
          << "n=" << n << " p=" << p;
    }
  }
}

TEST(HeroVillain, IncrementalMatchesBinarySearchOnMediumInputs) {
  // Large enough that brute force is out of reach; the two independent
  // algorithms must agree.
  for (const int p : {997, 5003}) {
    for (const int n : {50, 100, 150}) {
      const auto f = fMatrix(n, p);
      EXPECT_EQ(heroVillainValueIncremental(f, n, n), heroVillainValue(f, n, n))
          << "n=" << n << " p=" << p;
    }
  }
}

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------
void solveMain() {
  constexpr int kN = 611;
  constexpr int kP = 14411;
  const auto f = fMatrix(kN, kP);
  std::cout << "==> Hero-villain value for n = " << kN << ", p = " << kP
            << ": " << heroVillainValueIncremental(f, kN, kN) << std::endl;
}

void solveBonus() {
  constexpr int kMaxN = 999;  // 1 < n < 1000
  constexpr int kP = 17377;
  const auto f = fMatrix(kMaxN, kP);
  std::vector<int> value(kMaxN + 1, 0);
  std::atomic<int> nextN{2};
  const int numThreads = std::max(1, omp_get_max_threads());
  std::vector<std::thread> threads;
  for (int tid = 0; tid < numThreads; ++tid) {
    threads.emplace_back([&f, &value, &nextN] {
      while (true) {
        const int n = nextN.fetch_add(1);
        if (n > kMaxN) {
          break;
        }
        value[n] = heroVillainValueIncremental(f, kMaxN, n);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  int bestN = 2;
  for (int n = 2; n <= kMaxN; ++n) {
    if (value[n] > value[bestN]) {
      bestN = n;
    }
  }
  std::cout << "==> Bonus: optimal n for p = " << kP << " is " << bestN
            << " with hero-villain value " << value[bestN] << std::endl;
  for (int n = 2; n <= kMaxN; ++n) {
    if (n != bestN && value[n] == value[bestN]) {
      std::cout << "    (tie: n = " << n << ")" << std::endl;
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
  for (int i = 1; i < argc; ++i) {
    doSolve |= std::string(argv[i]) == "solve";
    doBonus |= std::string(argv[i]) == "bonus";
  }
  if (doSolve) {
    solveMain();
  }
  if (doBonus) {
    solveBonus();
  }
  return res;
}
