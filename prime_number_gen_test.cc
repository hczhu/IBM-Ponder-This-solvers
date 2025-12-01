#include "prime_number_gen.h"
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <vector>
#include <algorithm>

// Tests
TEST(PrimeNumberGenTest, SmallRange) {
    PrimeNumberGen pg(1, 20);
    std::vector<uint64_t> primes;
    for (uint64_t p : pg) {
        primes.push_back(p);
    }
    std::vector<uint64_t> expected = {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL};
    EXPECT_EQ(primes, expected);
}

TEST(PrimeNumberGenTest, RangeWithStart) {
    PrimeNumberGen pg(10, 30);
    std::vector<uint64_t> primes;
    for (uint64_t p : pg) {
        primes.push_back(p);
    }
    std::vector<uint64_t> expected = {11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL};
    EXPECT_EQ(primes, expected);
}

TEST(PrimeNumberGenTest, CountPrimes) {
    // Count primes less than 1000. There are 168 primes.
    PrimeNumberGen pg(1, 1000);
    int count = 0;
    for (auto p : pg) {
        (void)p;
        count++;
    }
    EXPECT_EQ(count, 168);
}

TEST(PrimeNumberGenTest, LargeRangeCount) {
    // Count primes less than 100,000. There are 9592 primes.
    PrimeNumberGen pg(1, 100000);
    int count = 0;
    for (auto p : pg) {
        (void)p;
        count++;
    }
    EXPECT_EQ(count, 9592);
}

// Benchmarks
static void BM_PrimeGenConstruction(benchmark::State& state) {
  for (auto _ : state) {
    PrimeNumberGen pg(1, 1'000'000'000ULL);
    benchmark::DoNotOptimize(pg);
  }
}
BENCHMARK(BM_PrimeGenConstruction)->Unit(benchmark::kMillisecond);

static void BM_PrimeGenConstructionLarge(benchmark::State& state) {
  for (auto _ : state) {
    PrimeNumberGen pg(1, 10'000'000'000ULL);
    benchmark::DoNotOptimize(pg);
  }
}
BENCHMARK(BM_PrimeGenConstructionLarge)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    
    if (ret == 0) {
        benchmark::Initialize(&argc, argv);
        benchmark::RunSpecifiedBenchmarks();
    }
    return ret;
}
