#pragma once

#include <cstdint>
#include <vector>

class PrimeNumberGen {
private:
  std::vector<uint64_t> notPrime_; // bit vector
  uint64_t low_, high_;


 // Helper methods for bit manipulation
 static constexpr uint64_t wordIndex(uint64_t n) { return n >> 6; }
 static constexpr uint64_t bitIndex(uint64_t n) { return n & 63; }
 void setNotPrime(uint64_t n) {
   notPrime_[wordIndex(n)] |= (1ULL << bitIndex(n));
 }

public:
  PrimeNumberGen(uint64_t low, uint64_t high);
  ~PrimeNumberGen();
  PrimeNumberGen(const PrimeNumberGen&) = delete;
  PrimeNumberGen& operator=(const PrimeNumberGen&) = delete;
  PrimeNumberGen(PrimeNumberGen&&) = delete;
  PrimeNumberGen& operator=(PrimeNumberGen&&) = delete;

  bool isNotPrime(uint64_t n) const {
    return (notPrime_[wordIndex(n)] >> bitIndex(n)) & 1;
  }
  bool isPrime(uint64_t n) const { return !isNotPrime(n); }

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
