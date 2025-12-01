#include "prime_number_gen.h"
#include <algorithm>
#include <glog/logging.h>

PrimeNumberGen::PrimeNumberGen(uint64_t low, uint64_t high) : low_(low), high_(high) {
  CHECK_GT(high, low);
  CHECK_GT(low, 0u);
  DLOG(INFO) << "Initing the prime number table of size " << high;
  notPrime_ = new bool[high + 1];
  std::fill(notPrime_, notPrime_ + high + 1, false);
  notPrime_[0] = true;
  notPrime_[1] = true;
  for (uint64_t p = 2; p * p <= high; ++p) {
    if (notPrime_[p]) {
      continue;
    }
    for (uint64_t n = p * p; n <= high; n += p) {
      notPrime_[n] = true;
    }
  }
  DLOG(INFO) << "Initialized the prime number table of size " << high;
}

PrimeNumberGen::~PrimeNumberGen() { delete[] notPrime_; }

uint64_t PrimeNumberGen::Itr::operator*() const {
  return ptr - base;
}

const PrimeNumberGen::Itr& PrimeNumberGen::Itr::operator++() {
  do {
    ++ptr;
  } while (ptr < end_ptr && *ptr);
  return *this;
}

bool PrimeNumberGen::Itr::operator==(const Itr rhs) const {
  return ptr == rhs.ptr && base == rhs.base;
}

PrimeNumberGen::Itr PrimeNumberGen::begin() const {
  Itr b{
      .ptr = notPrime_ + low_,
      .base = notPrime_,
      .end_ptr = notPrime_ + high_,
  };
  while (b.ptr < b.end_ptr && *b.ptr) {
    // Not a prime
    b.ptr++;
  }
  return b;
}

PrimeNumberGen::Itr PrimeNumberGen::end() const {
  return Itr{
    .ptr = notPrime_ + high_,
    .base = notPrime_,
    .end_ptr = notPrime_ + high_,
  };
}
