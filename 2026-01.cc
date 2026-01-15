/*
  Source code for the puzzle
  https://research.ibm.com/haifa/ponderthis/challenges/January2026.html

  The decimal notation string of the number 123 can be split in various ways: To 12 and 3 , to 1 and 23 , and to 123 . For each way to split the number, we add up the obtained parts. For 12 and 3 , we obtain the number 15 ; for 1 and 23 , we obtain the number 24 ; and for 123 , we obtain 123 itself. We denote this by A_{123}=\{15, 24, 123\} .

For a general natural number n , we define A_n similarly as the set of all natural numbers that can be obtained by splitting and adding the decimal notation of n . For example, A_{31658} = \{23, 32, 50, 68, 77, 95, 104, 176, 329, 374, 662, 689, 1661, 3173, 31658\}

Given a set A of integers and an integer n we use the standard notation n\cdot A\triangleq\{n\cdot x\ |\ x\in A\} (where \cdot is the usual multiplication of numbers).


Your goal: Find the sum of all the natural numbers x such that there exists 1\le n\le 10^{6} for which x\in n\cdot A_n and n\in A_x .

A bonus "*" will be given for finding the sum of all the natural numbers x such that there exists 1\le n\le 10^{7} for which x\in n\cdot A_n and n\in A_x .
*/

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <shared_mutex>
#include <thread>
#include <omp.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using Nat = uint64_t;

// Return true if f returns true for any value
bool genAn(Nat n, std::function<bool(Nat)> f) {
  int digits[20];
  const int nd = [&] {
    int nd = 0;
    for (; n > 0; n /= 10) {
      digits[nd++] = n % 10;
    }
    return nd;
  }();
  std::reverse(digits, digits + nd);
  for (int mask = (1<< (nd - 1 )); mask > 0; --mask) {
    Nat total = 0, sum = 0;
    for (int i = 0, mk = mask - 1; i < nd; ++i, mk >>= 1) {
      sum = sum * 10 + digits[i];
      const auto gate = mk & 1;
      // If gate is 0:
      //  - add sum to total
      //  - reset sum to 0
      // If gate is 1:
      //  - do nothing 
      total += sum * (gate ^ 1);
      sum *= gate;
    }
    assert(sum == 0);
    if (f(total)) {
      return true;
    }
  }
  return false;
}

bool AxContains(Nat x, Nat n) {
  auto fastPath = [=] {
    if ((x % n) != 0) {
      return false;
    }
    switch (x / n) {
      case 1:
      case 10:
      case 100:
      case 1'000:
      case 10'000:
      case 100'000:
      case 1'000'000:
      case 10'000'000:
        return true;
    default:
      return false;
    }
  }();
  if (fastPath) {
    return true;
  }
  return genAn(x, [&] (Nat y) { return y == n; });
}

TEST(smallTest, Basic) {
  std::unordered_set<Nat> result;
  genAn(123, [&] (Nat x) { result.insert(x); return false; });
  EXPECT_EQ(result, std::unordered_set<Nat>({6, 15, 24, 123}));
  result.clear();
  genAn(31658, [&] (Nat x) { result.insert(x); return false; });
  EXPECT_EQ(result, std::unordered_set<Nat>({23, 32, 50, 68, 77, 95, 104, 176, 329, 374, 662, 689, 1661, 3173, 31658}));

  EXPECT_TRUE(AxContains(123, 6));
  EXPECT_TRUE(AxContains(123, 15));
  EXPECT_TRUE(AxContains(123, 24));
  EXPECT_TRUE(AxContains(123, 123));
  EXPECT_FALSE(AxContains(123, 124));
  EXPECT_FALSE(AxContains(123, 125));

  EXPECT_TRUE(AxContains(31658, 23));
  EXPECT_TRUE(AxContains(31658, 32));
  EXPECT_TRUE(AxContains(31658, 50));
  EXPECT_TRUE(AxContains(31658, 68));
  EXPECT_TRUE(AxContains(31658, 77));
  EXPECT_TRUE(AxContains(31658, 95));
  EXPECT_TRUE(AxContains(31658, 104));
  EXPECT_TRUE(AxContains(31658, 176));
  EXPECT_TRUE(AxContains(31658, 329));
  EXPECT_TRUE(AxContains(31658, 374));
  EXPECT_TRUE(AxContains(31658, 662));
  EXPECT_TRUE(AxContains(31658, 689));
  EXPECT_TRUE(AxContains(31658, 1661));
  EXPECT_TRUE(AxContains(31658, 3173));
  EXPECT_TRUE(AxContains(31658, 31658));
  EXPECT_FALSE(AxContains(31658, 31659));
  EXPECT_FALSE(AxContains(31658, 31657));
}

Nat solve(int N, int numThreads) {
  LOG(INFO) << "Using " << numThreads << " threads";
  std::vector<std::vector<Nat>> answers(numThreads);
  std::vector<std::thread> threads;
  for (int id = 0; id < numThreads; ++id) {
    threads.push_back(std::thread([N, id, numThreads, &answers] {
      for (int n=id+1; n <= N; n += numThreads) {
        auto cb = [&](Nat x) {
          x *= n;
          if (AxContains(x, n)) {
            answers[id].push_back(x);
            LOG_IF(INFO, (answers[id].size() % 1729) == 0) << "Found " << answers[id].size() << "-th answer for thread " << id << ": " << x;
          }
          return false;
        };
        genAn(n, cb);
      }
    }));
  }
  for (int id = 0; id < numThreads; ++id) {
    threads[id].join();
    LOG(INFO) << "Thread " << id << " finished with " << answers[id].size() << " answers";
  }
  Nat total = 0;
  std::unordered_set<Nat> unique;
  for (auto& s : answers) {
    for (auto x : s) {
      if (unique.count(x) == 0) {
        total += x;
        unique.insert(x);
      }
    }
  }
  LOG(INFO) << "There are " << unique.size() << " unique answers";
  return total;
}


struct Accumulator {
  std::unordered_set<Nat> seen;
  bool operator() (Nat x) {
    seen.insert(x);
    return false;
  }
};

Nat bruteForce(int N) {
  std::unordered_set<Nat> unique;
  for (int n = 1; n <= N; ++n) {
    Accumulator an;
    genAn(n, [&an](Nat x) { return an(x); });
    for (auto xx : an.seen) {
      const auto x = xx * n;
      if (unique.count(x) == 0) {
        Accumulator ax;
        genAn(x, [&ax](Nat y) { return ax(y); });
        if (ax.seen.count(n)) {
          unique.insert(x);
        }
      }
    }
  }
  Nat total = 0;
  for (auto x : unique) {
    total += x;
  }
  LOG(INFO) << "There are " << unique.size() << " unique answers";
  return total;
}

TEST(solveTest, Basic) {
  EXPECT_EQ(bruteForce(1'000), solve(1'000, 31));
  EXPECT_EQ(bruteForce(10'000), solve(10'000, 31));
  EXPECT_EQ(bruteForce(100'000), solve(100'000, 31));
}
    
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();
  if (argc <= 1 || std::string(argv[1]) != "solve") {
    return res;
  }
  int numThreads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    LOG(INFO) << "threads = " << omp_get_num_threads();
    numThreads = omp_get_num_threads();
  }
  std::cout << "==> " << solve(1'000'000, numThreads) << std::endl;
  std::cout << "==> " << solve(10'000'000, numThreads) << std::endl;
  return res;
}
