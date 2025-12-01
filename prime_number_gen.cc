#include "prime_number_gen.h"
#include <cmath>
#include <cstring>
#include <glog/logging.h>

PrimeNumberGen::PrimeNumberGen(uint64_t low, uint64_t high)
    : low_(low), high_(high) {
  CHECK_GT(high, low);
  CHECK_GT(low, 0u);
  DLOG(INFO) << "Initing the prime number table of size " << high;

  // Allocate bit vector: (high + 2) bits for sentinel, stored in uint64_t words
  arraySize_ = high / 64 + 2;
  // Use a bit vector to reduce memory usage. Memory access is the bottleneck.
  notPrime_ = new uint64_t[arraySize_]();

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
  for (uint64_t p = 2; p <= sqrtHigh; ++p) {
    if (isNotPrime(p)) {
      continue;
    }
    for (uint64_t n = p * p; n <= high; n += p) {
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
