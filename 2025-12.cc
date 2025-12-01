/*
  Source code for the puzzle
  https://research.ibm.com/haifa/ponderthis/challenges/December2025.html

  Given a natural number n, take the first n odd primes (3, 5, 7, 11, ...)
  and the first n positive even integers (2, 4, ..., 2n).

  Compute all possible pairwise sums between these two sets (n^2 sums in total,
  all odd). f(n) = the number of these sums that are prime.

  Example: f(5) = 16
*/

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "prime_number_gen.h"
#include <cmath>

// Get the first n primes (3, 5, 7, 11, ...)
std::vector<uint64_t> getFirstNOddPrimes(int n) {
  std::vector<uint64_t> result;
  result.reserve(n);

  // The nth prime is approximately n * ln(n) for large n
  uint64_t upperBound = std::max(static_cast<uint64_t>(100),
                                 static_cast<uint64_t>(n * std::log(n) * 2));

  PrimeNumberGen gen(3, upperBound);
  for (uint64_t p : gen) {
    result.push_back(p);
    if (result.size() == static_cast<size_t>(n))
      break;
  }

  // If we didn't get enough primes, increase the bound and try again
  while (result.size() < static_cast<size_t>(n)) {
    upperBound *= 2;
    result.clear();
    PrimeNumberGen gen2(3, upperBound);
    for (uint64_t p : gen2) {
      result.push_back(p);
      if (result.size() == static_cast<size_t>(n))
        break;
    }
  }

  return result;
}

// Get the first n positive even integers (2, 4, 6, ..., 2n)
std::vector<uint64_t> getFirstNEvenIntegers(int n) {
  std::vector<uint64_t> result;
  result.reserve(n);
  for (int i = 1; i <= n; ++i) {
    result.push_back(2 * i);
  }
  return result;
}

uint64_t bruteForce(uint64_t n) {
  auto oddPrimes = getFirstNOddPrimes(n);
  auto evenInts = getFirstNEvenIntegers(n);

  // Find the maximum possible sum to create prime sieve
  uint64_t maxSum = oddPrimes.back() + evenInts.back();

  // Create prime sieve up to maxSum
  PrimeNumberGen primeGen(2, maxSum + 1);

  // Count prime sums
  uint64_t count = 0;
  for (uint64_t p : oddPrimes) {
    for (uint64_t e : evenInts) {
      uint64_t sum = p + e;
      if (primeGen.isPrime(sum)) {
        ++count;
      }
    }
  }

  return count;
}

TEST(PuzzleTest, FirstOddPrimes) {
  auto primes = getFirstNOddPrimes(5);
  ASSERT_EQ(primes.size(), 5);
  EXPECT_EQ(primes[0], 3);
  EXPECT_EQ(primes[1], 5);
  EXPECT_EQ(primes[2], 7);
  EXPECT_EQ(primes[3], 11);
  EXPECT_EQ(primes[4], 13);
}

TEST(PuzzleTest, FirstEvenIntegers) {
  auto evens = getFirstNEvenIntegers(5);
  ASSERT_EQ(evens.size(), 5);
  EXPECT_EQ(evens[0], 2);
  EXPECT_EQ(evens[1], 4);
  EXPECT_EQ(evens[2], 6);
  EXPECT_EQ(evens[3], 8);
  EXPECT_EQ(evens[4], 10);
}

TEST(PuzzleTest, F5Equals16) {
  // Given in the problem: f(5) = 16
  EXPECT_EQ(bruteForce(5), 16);
}

TEST(PuzzleTest, F1) {
  // n=1: odd primes = {3}, evens = {2}
  // sums = {5}, which is prime
  // f(1) = 1
  EXPECT_EQ(bruteForce(1), 1);
}

TEST(PuzzleTest, F2) {
  // n=2: odd primes = {3, 5}, evens = {2, 4}
  // sums: 3+2=5 (prime), 3+4=7 (prime), 5+2=7 (prime), 5+4=9 (not prime)
  // f(2) = 3
  EXPECT_EQ(bruteForce(2), 3);
}

TEST(PuzzleTest, F3) {
  // n=3: odd primes = {3, 5, 7}, evens = {2, 4, 6}
  // sums:
  // 3+2=5 (prime), 3+4=7 (prime), 3+6=9 (not)
  // 5+2=7 (prime), 5+4=9 (not), 5+6=11 (prime)
  // 7+2=9 (not), 7+4=11 (prime), 7+6=13 (prime)
  // f(3) = 6
  EXPECT_EQ(bruteForce(3), 6);
}

TEST(PuzzleTest, F4) {
  // n=4: odd primes = {3, 5, 7, 11}, evens = {2, 4, 6, 8}
  // f(4) = 11 (verified by running the program)
  EXPECT_EQ(bruteForce(4), 11);
}

TEST(PuzzleTest, SmallValues) {
  // Print f(n) for small values to see the pattern
  for (int n = 1; n <= 10; ++n) {
    uint64_t fn = bruteForce(n);
    LOG(INFO) << "f(" << n << ") = " << fn;
  }
}

uint64_t solve(uint64_t n) {
  const auto smallPrimes = getFirstNOddPrimes(n);
  uint64_t maxSum = smallPrimes.back() + n * 2;
  PrimeNumberGen primeGen(3, maxSum);
  std::vector<uint64_t> primes;
  for (uint64_t p : primeGen) {
    primes.push_back(p);
  }
  LOG(INFO) << "There are " << primes.size() << " primes up to " << maxSum;
  uint64_t count = 0;
  int l = 0;
  for (int i = 1; i < primes.size(); ++i) {
    auto p = primes[i];
    while (l < n && p - primes[l] > n * 2) {
      ++l;
    }
    count += (std::min(uint64_t(i), n) - l);
    DLOG(INFO) << "Processing prime " << p << " with l = " << l
               << " i - l = " << (i - l);
  }
  return count;
}

TEST(PuzzleTest, SolveF5) { EXPECT_EQ(solve(5), 16); }

TEST(PuzzleTest, SolveF1000) { EXPECT_EQ(solve(1'000), bruteForce(1'000)); }
TEST(PuzzleTest, SolveF2000) { EXPECT_EQ(solve(2'000), bruteForce(2'000)); }
TEST(PuzzleTest, SolveF3000) { EXPECT_EQ(solve(3'000), bruteForce(3'000)); }
TEST(PuzzleTest, SolveF4000) { EXPECT_EQ(solve(4'000), bruteForce(4'000)); }
TEST(PuzzleTest, SolveF8000) { EXPECT_EQ(solve(8'000), bruteForce(8'000)); }
TEST(PuzzleTest, SolveF8100) { EXPECT_EQ(solve(8'100), bruteForce(8'100)); }
TEST(PuzzleTest, SolveF9100) { EXPECT_EQ(solve(9'100), bruteForce(9'100)); }
TEST(PuzzleTest, SolveF20001) { EXPECT_EQ(solve(20'001), bruteForce(20'001)); }
TEST(PuzzleTest, SolveF200010) {
  EXPECT_EQ(solve(200'010), bruteForce(200'010));
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();
  std::cout << solve(100'000'000) << std::endl;
  std::cout << solve(1'000'000'000) << std::endl;
  return res;
}
