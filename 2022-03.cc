/*
  Source code for the puzzle
  https://research.ibm.com/haifa/ponderthis/challenges/March2022.html
*/
#include <algorithm>
#include <atomic>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <omp.h>
#include <algorithm>

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

#define POW(a, b) Pow<(b), (a)>::v

// See: https://www.ibm.com/docs/en/zos/2.4.0?topic=only-variadic-templates-c11
template<unsigned head, unsigned... tails>
struct ColorCode {
  static constexpr unsigned v = (ColorCode<tails...>::v << 2) ^ (head);
};

template<unsigned head>
struct ColorCode<head> {
  static constexpr unsigned v = head;
};

#define CC(colors...) (ColorCode<colors>::v)

class PrimeNumberGen {
private:
  bool *notPrime_ = nullptr;
  int low_, high_;

public:
  PrimeNumberGen(int low, int high) : low_(low), high_(high) {
    CHECK_GT(high, low);
    CHECK_GT(low, 0);
    LOG(INFO) << "Initing the prime number table of size " << high;
    notPrime_ = new bool[high + 1];
    std::fill(notPrime_, notPrime_ + high + 1, false);
    for (int p = 2; p * p < high; ++p) {
      if (notPrime_[p]) {
        continue;
      }
      for (int n = p * p; n < high; n += p) {
        notPrime_[n] = true;
      }
    }
    LOG(INFO) << "Initialized the prime number table of size " << high;
  }

  ~PrimeNumberGen() { delete[] notPrime_; }

  struct Itr {
    bool* ptr = nullptr;
    const bool* base = nullptr;
    int operator*() const {
      return ptr - base;
    }
    const Itr& operator++() {
      while (*++ptr);
      return *this;
    }

    bool operator==(const Itr rhs) const {
      return ptr == rhs.ptr && base == rhs.base;
    }
  };

  Itr begin() const {
    Itr b{
        .ptr = notPrime_ + low_,
        .base = notPrime_,
    };
    while (*b.ptr) {
      // Not a prime
      ++b;
    }
    return b;
  }

  Itr end() const {
    return Itr{
      .ptr = notPrime_ + high_,
      .base = notPrime_,
    };
  }
};

constexpr int GRAY = 0;
constexpr int YELLOW = 1;
constexpr int GREEN = 2;
using BinAndBM = std::pair<int, int>;

BinAndBM toBin(int n, int L) {
  int bin = 0;
  int bm = 0;
  for (int i = 0; i < L; ++i) {
    bin ^= ((n % 10) << (4 * i));
    bm |= 1 << (n % 10);

    n /= 10;
  }
  return BinAndBM(bin, bm);
}

int getColorsCode(int L, BinAndBM guess, BinAndBM solution, int prevSolution,
                  int prevColor) {
  if (0 == (guess.second & solution.second)) {
    // All gray.
    return 0;
  }
  int ret = 0;
  int gb = guess.first;
  int sb = solution.first;
  // A shortcut
  if (0 && (prevSolution ^ sb) < 10) {
    // Only differ at the last digit.
    int g = gb & 0xf;
    int s = sb & 0xf;
    int color = (prevColor & (~0xff));
    if (g == s) {
      return color ^ GREEN;
    }
    if ((1 << g) & solution.second) {
      return color ^ YELLOW;
    }
    return color ^ GRAY;
  }
  for (int i = 0; i < L; ++i) {
    int g = gb & 0xf;
    gb >>= 4;
    int s = sb & 0xf;
    sb >>= 4;
    int color = GRAY;
    if (g == s) {
      color = GREEN;
    } else if ((1 << g) & solution.second) {
      color = YELLOW;
    }
    // LOG(INFO) << g << " vs. " << s << " with color: " << color;
    ret ^= color << (2 * i);
  }
  return ret;
}

int getColorsCode(int L, int guess, int solution) {
  return getColorsCode(L, toBin(guess, L), toBin(solution, L), 0, 0);
}

std::string toColorSeq(int color, int L) {
  std::string ret;
  const std::vector<std::string> colorStr = {
    "gray",
    "yellow",
    "green",
  };
  for (int i = L - 1; i >= 0; --i) {
    ret += " " + colorStr[(color >> (2 * i)) & 3];
  }
  return ret;
}

int remainingSolutions(int L, int guess, int solution, const std::vector<int>& primes) {
  int ret = 0;
  const int targetColor = getColorsCode(L, guess, solution);
  LOG(INFO) << "target " << toColorSeq(targetColor, L) << " with " << solution;
  int prevS = 0; 
  int prevColor = 0;
  int prevBinS = 0;
  const auto binGuess = toBin(guess, L);
  for (auto s : primes) {
    const auto binS = toBin(s, L);
    const auto color = getColorsCode(L, binGuess, binS, prevBinS, prevColor);
    LOG(INFO) << guess << " " << s << " " << prevS << " " << toColorSeq(prevColor, L) << " = " << toColorSeq(color, L);
    if (color == targetColor) {
      ++ret;
    }
    prevS = s;
    prevBinS = binS.first;
    prevColor = color;
  }
  return ret;
}

TEST(getColor4773Test, Basic) {
  const int L = 4;
  int prevSolution = 0;
  int prevColor = 0;
  const auto binS = toBin(4733, L);
  {
    int p = 1009;
    const auto binP = toBin(p, L);
    prevColor = getColorsCode(4, binP, binS, prevSolution, prevColor);
    prevSolution = binS.first;
    CHECK_EQ(toColorSeq(prevColor, L), " gray gray gray gray");
  }
  {
    int p = 1013;
    const auto binP = toBin(p, L);
    prevColor = getColorsCode(4, binP, binS, prevSolution, prevColor);
    CHECK_EQ(toColorSeq(prevColor, L), " gray gray gray green");
  }
  {
    CHECK_EQ(
        toColorSeq(getColorsCode(4, toBin(3637, L), toBin(4733, L), 0, 0), L),
        " yellow gray green yellow");
  }
}

TEST(PrimeGen, Basic) {
  EXPECT_EQ(0, getColorsCode(4, toBin(1009, 4), toBin(4733, 4), 0, 0));
  static_assert(POW(10, 3) == 1000);
  PrimeNumberGen pg(POW(10, 1), POW(10, 2));
  EXPECT_EQ(11, *pg.begin());
  EXPECT_NE(pg.end(), pg.begin());
  std::vector<int> answers = {
      29, 23, 19, 17, 13, 11,
  };
  for (int p : pg) {
    EXPECT_EQ(answers.back(), p);
    answers.pop_back();
    if (answers.empty()) {
      break;
    }
  }

  auto countPrimeN = [](int low, int high) {
    int cnt = 0;
    for (auto p : PrimeNumberGen(low, high)) {
      ++cnt;
    }
    return cnt;
  };
  EXPECT_EQ(21, countPrimeN(POW(10, 1), POW(10, 2)));
  EXPECT_EQ(8'363, countPrimeN(POW(10, 4), POW(10, 5)));
  EXPECT_EQ(586'081, countPrimeN(POW(10, 6), POW(10, 7)));

  EXPECT_EQ(CC(YELLOW, GRAY, GREEN), (2<<4) + (1));

  {
    auto pr = toBin(610, 3);
    CHECK_EQ(pr.first, (6 << 8) + (1 << 4));
    CHECK_EQ(pr.second, (1 << 6) + (1 << 1) + (1 << 0));
  }
  {
    auto pr = toBin(1733, 4);
    CHECK_EQ(pr.first, (1 << 12) + (7 << 8) + (3 << 4) + (3 << 0));
    CHECK_EQ(pr.second, (1 << 1) + (1 << 7) + (1 << 3));
  }

  const auto expectedColor = CC(YELLOW, GREEN, GRAY, YELLOW);
  const std::vector<int> numbers = {
    1733, 2731, 4733, 7039, 7331, 7333, 7433, 7933, 8731, 9733, 9739
  };
  for (auto p : numbers) {
    CHECK_EQ(getColorsCode(4, 3637, p), expectedColor) << " wrong for " << p;
  }

  PrimeNumberGen pg4(1'000, 10'000);
  std::vector<int> primes4;
  for (auto p : pg4) {
    primes4.push_back(p);
  }
  EXPECT_EQ(1229 - 168, primes4.size());
  EXPECT_EQ(11, remainingSolutions(4, 3637, 4733, primes4));
  // EXPECT_EQ(8, remainingSolutions(4, 8731, 4733, primes4));
}

uint64_t remainingSolutionsForAll(int guessIdx,
                                  const std::vector<BinAndBM> &primes,
                                  int L,
                                  uint64_t upperBound) {
  int colorToCount[1<<14] = {
    0
  };
  CHECK_EQ(colorToCount[(1<<14) - 1], 0);
  const auto guess = primes[guessIdx];
  uint64_t ret = primes.size();
  int prevSolution = 0;
  int prevColor = 0;
  for (auto p : primes) {
    const auto color = getColorsCode(L, guess, p, prevSolution, prevColor);
    prevSolution = p.first;
    prevColor = color;
    auto &cnt = colorToCount[color];
    ret += (cnt << 1);
    ++cnt;
    if (ret > upperBound) {
      break;
    }
  }
  return ret;
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // omp_set_num_threads(8);

  auto ret = RUN_ALL_TESTS();
  CHECK_EQ(ret, 0) << "RUN_ALL_TESTS returns: " << ret;

  PrimeNumberGen pg5(10'000, 100'000);
  PrimeNumberGen pg7(1'000'000, 10'000'000);

  auto solver =[](const PrimeNumberGen& pg, int L) {
    std::vector<int> rawPrimes;
    std::vector<BinAndBM> primes;
    for (auto p : pg) {
      rawPrimes.push_back(p);
      primes.push_back(toBin(p, L));
    }
    // Pass in an env OMP_NUM_THREADS=8
    LOG(INFO) << "There are " << primes.size() << " prime numbers.";
    // std::random_shuffle(primes.begin(), primes.end());
    std::vector<uint64_t> sums(primes.size());
    const int numP = primes.size();
    bool threadsOnce[100] = {
      false,
    };
    // compare_exchange_strong( T& expected, T desired )
    std::atomic<uint64_t> minExp(std::numeric_limits<uint64_t>::max());
    #pragma omp parallel for
    for (int i = 0; i < numP; ++i) {
      const auto sum = remainingSolutionsForAll(i, primes, L, minExp);
      uint64_t updatedMinExp = minExp.load();
      while (sum < updatedMinExp &&
             !minExp.compare_exchange_strong(updatedMinExp, sum))
        ;
      sums[i] = sum;
      if (0 == (i % 1000)) {
        LOG(INFO) << "Got result for the " << i
                  << "-th prime with current min: " << minExp;
      }
    }
    uint64_t minExpV = minExp;
    LOG(INFO) << "Min expectation: " << minExpV;
    for (int i = 0; i < numP; ++i) {
      CHECK_LE(minExpV, sums[i]);
      if (minExpV == sums[i]) {
        std::cout << "=== " << rawPrimes[i] << " : " << 1.0 * minExpV / numP << std::endl;
      }
    }
  };

  solver(pg5, 5);
  solver(pg7, 7);
  return 0;
}
