/*
  Source code for the puzzle
  https://research.ibm.com/haifa/ponderthis/challenges/January2022.html
*/
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

template<typename T>
class _DisplayType;

template<typename T>
void _displayType(T&& t);

#define PEEK(x) LOG(INFO) << #x << ": [" << (x) << "]"

#define BT(n) (1<<(n))

template<int P, int B>
struct Pow {
  static constexpr uint64_t v = Pow<P - 1, B>::v * B;
};

template<int B>
struct Pow<0, B> {
  static constexpr uint64_t v = 1;
};

constexpr int N = 8;
constexpr int D = 6;
constexpr int W = 10;
constexpr int P = Pow<D, W>::v;

// See: https://www.ibm.com/docs/en/zos/2.4.0?topic=only-variadic-templates-c11
template<unsigned head, unsigned... tails>
struct BitMask {
  static constexpr uint64_t v = (uint64_t(1) << (head)) | BitMask<tails...>::v;
};

template<unsigned head>
struct BitMask<head> {
  static constexpr uint64_t v = (uint64_t(1) << (head));
};

#define BM(bits...) (BitMask<bits>::v)

bool notPrime[P];
int bitMaskToEdgeCnt[BT(W)][W][W];

using IntPair = std::pair<int, int>;

int numToBitMask(int num, std::vector<IntPair>& ip) {
  int ret = 0;
  int prevDigit = -1;
  ip.clear();
  while (num > 0) {
    int d = num % 10;
    num /= 10;
    if (prevDigit >= 0) {
      ip.push_back(IntPair(prevDigit, d));
    }
    prevDigit = d;
    int b = 1 << d;
    if (b & ret) {
      // This is a duplicate digit.
      return 0;
    }
    ret ^= b;
  }
  return ret;
}


void initPrimeNumberTable(int d) {
  LOG(INFO) << "Initing the prime number table.";
  for (int p = 2; p * p < P; ++p) {
    if (notPrime[p]) {
      continue;
    }
    for (int n = p * p; n < P; n += p) {
      notPrime[n] = true;
    }
  }
  memset(bitMaskToEdgeCnt, 0, sizeof bitMaskToEdgeCnt);

  int primeCnt = 0;
  int edgeCnt = 0;
  int wd = 1;
  for (int i = 0; i < d; ++i) {
    wd *= W;
  }
  for (int p = wd / W; p < wd; ++p) {
    if (notPrime[p]) {
      continue;
    }
    std::vector<IntPair> ip;
    if (auto bm = numToBitMask(p, ip)) {
      if (ip.size() + 1 != d) {
        continue;
      }
      ++primeCnt; 
      int m = (((1<<W) - 1) ^ bm);
      int sm = m;
      do {
        auto bitMask = bitMaskToEdgeCnt[sm ^ bm];
        sm = (sm - 1) & m;
        for (auto&& [prev, next] : ip) {
          CHECK_NE(prev, next);
          ++bitMask[prev][next];
          ++bitMask[next][prev];
          ++edgeCnt;
        }
      } while (sm > 0);
    }
  }

  LOG(INFO) << "Finished initing the prime number table with " << primeCnt
            << " prime numbers and " << edgeCnt << " edges.";
}

int scoreCircle(const std::vector<int>& circle, int d) {
  const int n = circle.size();
  int bm = 0;
  for (auto d : circle) {
    bm ^= (1<<d);
  }
  auto edgeCnt = bitMaskToEdgeCnt[bm];
  int ret = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      ret += std::min(j - i, n - (j - i)) * edgeCnt[circle[i]][circle[j]];
    }
  }
  return ret;
}

int minScore = 10000000;
std::vector<int> minCircle;

int maxScore = -1;
std::vector<int> maxCircle;

void doSearch(int n, int circle[], int p, int bm, int score, const int edgeCnt[][W]) {
  if (bm == 0) {
    CHECK_EQ(n, p);
    if (score < minScore) {
      minScore = score;
      minCircle = std::vector<int>(circle, circle + n);
      CHECK_EQ(minCircle.size(), n);
    }

    if (score > maxScore) {
      maxScore = score;
      maxCircle = std::vector<int>(circle, circle + n);
    }
    return;
  }

  if (p >= N) {
    return; // Safety check to prevent out-of-bounds access
  }

  for (int d = 0; d < W; ++d) {
    if (((1<<d) & bm) == 0) {
      continue;
    }
    circle[p] = d;
    int addedScore = 0;
    for (int i = 0; i < p; ++i) {
      addedScore += edgeCnt[circle[i]][circle[p]] * std::min(p - i, n - (p - i));
    }
    doSearch(n, circle, p + 1, bm ^ (1<<d), addedScore + score, edgeCnt);
  }
}

void solve(int n, int d) {
  initPrimeNumberTable(d);
  std::vector<int> goodMb;
  for (int mb = 1; mb < (1<<W); ++mb) {
    if (__builtin_popcount(mb) == n) {
      goodMb.push_back(mb);
    }
  }
  LOG(INFO) << "There are " << goodMb.size() << " candidate selections.";

  minScore = 10000000;
  maxScore = -1;
  // #pragma omp parallel for
  for (auto mb : goodMb) {
    // LOG(INFO) << "Searching for selection bits: " << mb;
    int circle[N];
    circle[0] = __builtin_ffs(mb) - 1;
    doSearch(n, circle, 1, mb ^ (1<<circle[0]), 0, bitMaskToEdgeCnt[mb]);
  }
}


TEST(ScoringTest, Basic) {
  EXPECT_EQ(sizeof bitMaskToEdgeCnt,
            BM(W) * W * W * sizeof bitMaskToEdgeCnt[0][0][0]);
  EXPECT_EQ((1 << 4) + (1 << 7) + (1 << 3) + (1 << 6) + (1 << 2) + (1 << 0) +
                (1 << 1),
            BM(4, 7, 3, 6, 2, 0, 1, 1));

  initPrimeNumberTable(5);
  EXPECT_FALSE(notPrime[24103]);
  std::vector<IntPair> ip;
  int bm = numToBitMask(24103, ip);
  EXPECT_EQ(BM(2, 4, 1, 0, 3), bm);
  EXPECT_EQ(4, ip.size());

  EXPECT_EQ(1882, scoreCircle( { 4, 7, 3, 6, 2, 0, 1, }, 5));

  EXPECT_EQ(446, scoreCircle({0, 1, 4, 5, 8, 2, 6, }, 5));
  EXPECT_EQ(3051, scoreCircle({1, 2, 5, 4, 9, 3, 7}, 5));

  solve(3, 2);
  EXPECT_EQ(0, minScore);
  EXPECT_EQ(3, minCircle.size());

  EXPECT_EQ(6, maxScore);
  EXPECT_EQ(3, maxCircle.size());
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  auto ret = RUN_ALL_TESTS();
  LOG(INFO) << "RUN_ALL_TESTS returns: " << ret;
  auto solveOne = [&](int n, int d) {
    solve(n, d);
    std::cout << "Solved for n = " << n << " d = " << d << std::endl;

    std::cout << "Max score: " << maxScore << " with circle: [";
    std::ranges::copy(maxCircle, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << "]" << std::endl;

    std::cout << "Min score: " << minScore << " with circle: [";
    std::ranges::copy(minCircle, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << "]" << std::endl;

  };
  solveOne(7, 5);
  solveOne(8, 6);
  return ret;
}

