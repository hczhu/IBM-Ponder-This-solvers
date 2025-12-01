#include "prime_number_gen.h"
#include <algorithm>
#include <glog/logging.h>

PrimeNumberGen::PrimeNumberGen(uint64_t low, uint64_t high) : low_(low), high_(high) {
  CHECK_GT(high, low);
  CHECK_GT(low, 0u);
  DLOG(INFO) << "Initing the prime number table of size " << high;

  // Allocate bit vector: (high + 1) bits, stored in uint64_t words
  arraySize_ = (high + 64) / 64; // ceiling division
  notPrime_ = new uint64_t[arraySize_];
  std::fill(notPrime_, notPrime_ + arraySize_, 0ULL);

  // 0 and 1 are not prime
  setNotPrime(0);
  setNotPrime(1);

  for (uint64_t p = 2; p * p <= high; ++p) {
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
  do {
    ++current;
  } while (current < end_val && gen->isNotPrime(current));
  return *this;
}

bool PrimeNumberGen::Itr::operator==(const Itr rhs) const {
  return current == rhs.current && gen == rhs.gen;
}

PrimeNumberGen::Itr PrimeNumberGen::begin() const {
  Itr b{
      .gen = this,
      .current = low_,
      .end_val = high_,
  };
  while (b.current < b.end_val && isNotPrime(b.current)) {
    // Not a prime
    b.current++;
  }
  return b;
}

PrimeNumberGen::Itr PrimeNumberGen::end() const {
  return Itr{
      .gen = this,
      .current = high_,
      .end_val = high_,
  };
}
