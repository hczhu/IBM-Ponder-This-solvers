/*
  Source code for the puzzle
  https://research.ibm.com/blog/ponder-this-july-2026

  For 1 <= a,b <= n, start x_0=0 and iterate

      x_{k+1} = x_k^2 + a*x_k + b (mod p).

  The weight f(a,b) is the first k for which x_k has appeared before.  We seek
  a permutation pi maximizing min_a f(a,pi(a)): the bottleneck perfect-matching
  value of the n by n weight matrix.

  Orbit table
  -----------
  A timestamped array of p entries detects the first repeat with exactly one
  polynomial evaluation per orbit step.  The table is stored as uint16_t since
  f(a,b) <= p < 2^16 for both puzzle instances.  Rows are independent and are
  generated in parallel.  This costs O(n^2*p) time in the worst case (normally
  O(n^2*sqrt(p)) for rho-shaped orbits), O(n^2) table space, and O(p) scratch
  space per worker.

  Primary
  -------
  Edges whose weights are at least a threshold t form a bipartite graph.  The
  threshold is feasible exactly when the graph has a perfect matching.  This
  predicate is monotone, so an integer binary search plus Hopcroft-Karp finds
  the largest feasible t in O(log(p) * n^2 * sqrt(n)) worst-case time and O(n)
  extra space (the weight matrix is scanned directly, avoiding adjacency-list
  duplication).

  Bonus
  -----
  For a fixed threshold, grow the leading k by k graph from k=1 to N-1 while
  maintaining a maximum matching.  Adding one left and one right vertex can
  increase the maximum matching size by at most two, hence at most two new
  augmenting paths are needed at each k.  Adjacency rows are bitsets, making an
  alternating-path search word-parallel.  A second monotone binary search finds
  the greatest threshold attained by any 1 < k < N, then one final pass returns
  every k attaining it.  The threshold graph uses O(N^2/64) extra words.
  The worst-case matching work is O(log(p) * N^3/64), versus running an
  independent O(k^2*sqrt(k)) matching for every k at every threshold.
*/

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

class OrbitTable {
 public:
  OrbitTable(int n, int p) : n_(n), p_(p), weights_(size_t(n) * n) {
#pragma omp parallel
    {
      // Each worker owns its timestamps, so rows can be generated independently
      // without synchronization.  A worker handles fewer than n epochs, hence
      // int cannot wrap in either puzzle instance.
      std::vector<int> seen(p, 0);
      int epoch = 0;
#pragma omp for schedule(dynamic, 1)
      for (int a = 1; a <= n; ++a) {
        for (int b = 1; b <= n; ++b) {
          ++epoch;
          int x = 0;
          seen[x] = epoch;
          for (int step = 1;; ++step) {
            x = int((int64_t(x) * x + int64_t(a) * x + b) % p);
            if (seen[x] == epoch) {
              weights_[index(a - 1, b - 1)] = uint16_t(step);
              break;
            }
            seen[x] = epoch;
          }
        }
      }
    }
  }

  int size() const { return n_; }
  int modulus() const { return p_; }
  std::span<const uint16_t> row(int a) const {
    return {weights_.data() + size_t(a) * n_, size_t(n_)};
  }
  int operator()(int a, int b) const { return row(a)[b]; }

 private:
  size_t index(int a, int b) const { return size_t(a) * n_ + b; }

  int n_;
  int p_;
  std::vector<uint16_t> weights_;
};

// Hopcroft-Karp over the implicit threshold graph.  Scanning the compact
// weight matrix is faster and smaller than materializing a new adjacency list
// at every binary-search probe.
class HopcroftKarpThreshold {
 public:
  HopcroftKarpThreshold(const OrbitTable& table, int n, int threshold)
      : table_(table),
        n_(n),
        threshold_(threshold),
        match_left_(n, -1),
        match_right_(n, -1),
        distance_(n),
        next_right_(n) {}

  bool hasPerfectMatching() {
    int matching = 0;
    while (bfs()) {
      std::ranges::fill(next_right_, 0);
      for (const int left : std::views::iota(0, n_)) {
        if (match_left_[left] == -1 && dfs(left)) {
          ++matching;
        }
      }
      if (matching == n_) return true;
    }
    return false;
  }

 private:
  static constexpr int INF = std::numeric_limits<int>::max();

  bool bfs() {
    std::queue<int> queue;
    for (const int left : std::views::iota(0, n_)) {
      if (match_left_[left] == -1) {
        distance_[left] = 0;
        queue.push(left);
      } else {
        distance_[left] = INF;
      }
    }

    nil_distance_ = INF;
    while (!queue.empty()) {
      const int left = queue.front();
      queue.pop();
      if (distance_[left] + 1 > nil_distance_) continue;

      const auto weights = table_.row(left);
      for (const int right : std::views::iota(0, n_)) {
        if (weights[right] < threshold_) continue;
        const int next_left = match_right_[right];
        if (next_left == -1) {
          nil_distance_ = distance_[left] + 1;
        } else if (distance_[next_left] == INF) {
          distance_[next_left] = distance_[left] + 1;
          queue.push(next_left);
        }
      }
    }
    return nil_distance_ != INF;
  }

  bool dfs(int left) {
    const auto weights = table_.row(left);
    for (int& right = next_right_[left]; right < n_; ++right) {
      if (weights[right] < threshold_) continue;
      const int next_left = match_right_[right];
      if ((next_left == -1 && distance_[left] + 1 == nil_distance_) ||
          (next_left != -1 && distance_[next_left] == distance_[left] + 1 &&
           dfs(next_left))) {
        match_left_[left] = right;
        match_right_[right] = left;
        ++right;  // Do not reconsider this edge if this DFS frame is revisited.
        return true;
      }
    }
    distance_[left] = INF;
    return false;
  }

  const OrbitTable& table_;
  int n_;
  int threshold_;
  int nil_distance_ = INF;
  std::vector<int> match_left_;
  std::vector<int> match_right_;
  std::vector<int> distance_;
  std::vector<int> next_right_;
};

static bool feasibleExact(const OrbitTable& table, int n, int threshold) {
  return HopcroftKarpThreshold(table, n, threshold).hasPerfectMatching();
}

static int bottleneckValue(const OrbitTable& table, int n) {
  int feasible = 0;
  int infeasible = table.modulus() + 1;
  while (infeasible - feasible > 1) {
    const int middle = feasible + (infeasible - feasible) / 2;
    if (feasibleExact(table, n, middle)) {
      feasible = middle;
    } else {
      infeasible = middle;
    }
  }
  return feasible;
}

// Maintains maximum matchings of every leading k by k threshold graph.  The
// graph is represented as a flat array of row bitsets.
class PrefixMatcher {
 public:
  PrefixMatcher(const OrbitTable& table, int threshold)
      : n_(table.size()),
        words_((n_ + 63) / 64),
        adjacency_(size_t(n_) * words_, 0),
        match_left_(n_, -1),
        match_right_(n_, -1),
        seen_left_(n_),
        seen_right_(words_),
        parent_right_(n_),
        queue_(n_) {
    for (const int left : std::views::iota(0, n_)) {
      const auto weights = table.row(left);
      auto bits = adjacencyRow(left);
      for (const int right : std::views::iota(0, n_)) {
        if (weights[right] >= threshold) {
          bits[right / 64] |= uint64_t(1) << (right % 64);
        }
      }
    }
  }

  // Returns every 1 < k <= n_ whose leading k by k graph has a perfect
  // matching.  If stop_at_first is set, returns immediately after finding one.
  std::vector<int> perfectPrefixes(bool stop_at_first) {
    std::vector<int> result;
    int matching_size = 0;

    for (int k = 1; k <= n_; ++k) {
      // The previous prefix matching was maximum.  Adding one vertex on each
      // side raises the maximum size by at most two.
      for (int gained = 0; gained < 2 && augment(k); ++gained) {
        ++matching_size;
      }

      if (k > 1 && matching_size == k) {
        result.push_back(k);
        if (stop_at_first) return result;
      }
    }
    return result;
  }

 private:
  bool augment(int k) {
    std::fill(seen_left_.begin(), seen_left_.begin() + k, uint8_t(0));
    std::ranges::fill(seen_right_, uint64_t(0));

    int head = 0;
    int tail = 0;
    for (int left = 0; left < k; ++left) {
      if (match_left_[left] == -1) {
        seen_left_[left] = 1;
        queue_[tail++] = left;
      }
    }
    if (tail == 0) return false;

    const int active_words = (k + 63) / 64;
    const int tail_bits = k % 64;
    const uint64_t tail_mask =
        tail_bits == 0 ? ~uint64_t(0) : (uint64_t(1) << tail_bits) - 1;

    while (head < tail) {
      const int left = queue_[head++];
      const auto row = adjacencyRow(left);
      for (int word = 0; word < active_words; ++word) {
        uint64_t candidates = row[word] & ~seen_right_[word];
        if (word + 1 == active_words) candidates &= tail_mask;

        while (candidates != 0) {
          const int bit = std::countr_zero(candidates);
          const uint64_t bit_mask = uint64_t(1) << bit;
          candidates &= candidates - 1;
          seen_right_[word] |= bit_mask;

          const int right = word * 64 + bit;
          parent_right_[right] = left;
          const int next_left = match_right_[right];
          if (next_left == -1) {
            flipPath(right);
            return true;
          }
          if (!seen_left_[next_left]) {
            seen_left_[next_left] = 1;
            queue_[tail++] = next_left;
          }
        }
      }
    }
    return false;
  }

  void flipPath(int right) {
    while (right != -1) {
      const int left = parent_right_[right];
      const int previous_right = match_left_[left];
      match_left_[left] = right;
      match_right_[right] = left;
      right = previous_right;
    }
  }

  std::span<uint64_t> adjacencyRow(int left) {
    return {adjacency_.data() + size_t(left) * words_, size_t(words_)};
  }

  int n_;
  int words_;
  std::vector<uint64_t> adjacency_;
  std::vector<int> match_left_;
  std::vector<int> match_right_;
  std::vector<uint8_t> seen_left_;
  std::vector<uint64_t> seen_right_;
  std::vector<int> parent_right_;
  std::vector<int> queue_;
};

struct BonusResult {
  int value;
  std::vector<int> sizes;
};

static BonusResult bestPrefix(const OrbitTable& table) {
  int feasible = 0;
  int infeasible = table.modulus() + 1;
  while (infeasible - feasible > 1) {
    const int middle = feasible + (infeasible - feasible) / 2;
    PrefixMatcher matcher(table, middle);
    if (!matcher.perfectPrefixes(true).empty()) {
      feasible = middle;
    } else {
      infeasible = middle;
    }
  }

  PrefixMatcher matcher(table, feasible);
  return {.value = feasible, .sizes = matcher.perfectPrefixes(false)};
}

static void printRanges(std::span<const int> values) {
  for (size_t first = 0; first < values.size();) {
    size_t last = first;
    while (last + 1 < values.size() && values[last + 1] == values[last] + 1) {
      ++last;
    }
    if (first != 0) std::cout << ',';
    std::cout << values[first];
    if (last != first) std::cout << ".." << values[last];
    first = last + 1;
  }
}

static int directOrbitLength(int a, int b, int p) {
  std::vector<uint8_t> seen(p, 0);
  int x = 0;
  seen[x] = 1;
  for (int step = 1;; ++step) {
    x = int((int64_t(x) * x + int64_t(a) * x + b) % p);
    if (seen[x]) return step;
    seen[x] = 1;
  }
}

static int bruteBottleneck(const OrbitTable& table, int n) {
  std::vector<int> permutation(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  int best = 0;
  do {
    int value = table.modulus();
    for (int left = 0; left < n; ++left) {
      value = std::min(value, table(left, permutation[left]));
    }
    best = std::max(best, value);
  } while (std::ranges::next_permutation(permutation).found);
  return best;
}

TEST(July2026, OrbitLengthsFromExample) {
  EXPECT_EQ(directOrbitLength(1, 3, 101), 14);
  EXPECT_EQ(directOrbitLength(2, 1, 101), 18);
  EXPECT_EQ(directOrbitLength(3, 4, 101), 19);
  EXPECT_EQ(directOrbitLength(4, 2, 101), 22);
  EXPECT_EQ(directOrbitLength(5, 5, 101), 14);
}

TEST(July2026, GivenBottleneckExample) {
  const OrbitTable table(5, 101);
  EXPECT_EQ(bottleneckValue(table, 5), 14);
}

TEST(July2026, MatchesBruteForce) {
  const OrbitTable table(8, 101);
  for (int n = 1; n <= 8; ++n) {
    EXPECT_EQ(bottleneckValue(table, n), bruteBottleneck(table, n)) << "n=" << n;
  }
}

TEST(July2026, PrefixSearchMatchesIndependentAnswers) {
  const OrbitTable table(8, 101);
  const BonusResult result = bestPrefix(table);

  int expected_value = 0;
  std::vector<int> expected_sizes;
  for (int n = 2; n <= table.size(); ++n) {
    const int value = bruteBottleneck(table, n);
    if (value > expected_value) {
      expected_value = value;
      expected_sizes = {n};
    } else if (value == expected_value) {
      expected_sizes.push_back(n);
    }
  }
  EXPECT_EQ(result.value, expected_value);
  EXPECT_EQ(result.sizes, expected_sizes);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  testing::InitGoogleTest(&argc, argv);
  const int test_result = RUN_ALL_TESTS();
  if (test_result != 0 || argc <= 1 || std::string(argv[1]) != "solve") {
    return test_result;
  }

  int primary_value;
  {
    LOG(INFO) << "Building primary orbit table";
    const OrbitTable primary(611, 14411);
    primary_value = bottleneckValue(primary, 611);
  }  // Release the primary table before allocating the larger bonus table.
  std::cout << "Hero-villain value for n=611, p=14411: " << primary_value << '\n';

  LOG(INFO) << "Building bonus orbit table";
  const OrbitTable bonus(999, 17377);
  const BonusResult bonus_result = bestPrefix(bonus);
  std::cout << "Bonus maximum hero-villain value for p=17377: "
            << bonus_result.value << " at n=";
  printRanges(bonus_result.sizes);
  std::cout << '\n';
  return 0;
}
