#include "prime_number_gen.h"
#include <algorithm>
#include <cmath>
#include <glog/logging.h>

PrimeNumberGen::PrimeNumberGen(uint64_t low, uint64_t high)
    : low_(low), high_(high) {
  CHECK_GT(high, low);
  CHECK_GT(low, 0u);
  DLOG(INFO) << "Initing the prime number table of size " << high;

  arraySize_ = high / 64 + 2;
  notPrime_ = new uint64_t[arraySize_];
  memset(notPrime_, 0, arraySize_ * sizeof(uint64_t));

  // 0 and 1 are not prime
  setNotPrime(0);
  setNotPrime(1);

  const uint64_t sqrtHigh = [high]() {
    uint64_t res = static_cast<uint64_t>(std::sqrt(high)) + 2;
    while (res * res > high) {
      --res;
    }
    return res;
  }();
  const uint64_t sqrtSqrtHigh = static_cast<uint64_t>(std::sqrt(sqrtHigh)) + 1;
  for (uint64_t p = 2; p <= sqrtSqrtHigh; ++p) {
    if (isNotPrime(p)) {
      continue;
    }
    for (uint64_t n = p * p; n <= sqrtHigh; n += p) {
      setNotPrime(n);
    }
  }
#pragma omp parallel for schedule(static, 1)
  for (uint64_t p = 2; p <= sqrtHigh; ++p) {
    if (isNotPrime(p)) {
      continue;
    }
    for (uint64_t n = p * p; n <= high_; n += p) {
      setNotPrime(n);
    }
  }
  DLOG(INFO) << "Initialized the prime number table of size " << high;
}

PrimeNumberGen::~PrimeNumberGen() { delete[] notPrime_; }

uint64_t PrimeNumberGen::Itr::operator*() const { return current; }

const PrimeNumberGen::Itr& PrimeNumberGen::Itr::operator++() {
  // It will eventually hit the sentinel, high_ + 1.
  do {
    ++current;
  } while (gen->isNotPrime(current));
  return *this;
}

bool PrimeNumberGen::Itr::operator==(const Itr rhs) const {
  return current == rhs.current && gen == rhs.gen;
}

PrimeNumberGen::Itr PrimeNumberGen::begin() const {
  Itr b{
      .gen = this,
      .current = low_,
  };
  // It may hit the sentinel, high_ + 1.
  while (isNotPrime(b.current)) {
    b.current++;
  }
  return b;
}

PrimeNumberGen::Itr PrimeNumberGen::end() const {
  return Itr{
      .gen = this,
      // This is a sentinel value beyond the end and is declared a prime number
      // (not always true).
      .current = high_ + 1,
  };
}
