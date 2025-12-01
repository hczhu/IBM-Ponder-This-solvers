#pragma once

#include <cstdint>

class PrimeNumberGen {
private:
 bool* notPrime_ = nullptr;
 uint64_t low_, high_;

public:
  PrimeNumberGen(uint64_t low, uint64_t high);
  ~PrimeNumberGen();

  bool isNotPrime(uint64_t n) const { return notPrime_[n]; }

  struct Itr {
    const PrimeNumberGen* gen = nullptr;
    uint64_t current = 0;
    uint64_t operator*() const;
    const Itr& operator++();
    bool operator==(const Itr rhs) const;
  };

  Itr begin() const;
  Itr end() const;
};
