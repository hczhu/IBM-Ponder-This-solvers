#pragma once

#include <cstdint>

class PrimeNumberGen {
private:
  bool *notPrime_ = nullptr;
  uint64_t low_, high_;

public:
  PrimeNumberGen(uint64_t low, uint64_t high);
  ~PrimeNumberGen();

  struct Itr {
    bool* ptr = nullptr;
    const bool* base = nullptr;
    const bool* end_ptr = nullptr;
    uint64_t operator*() const;
    const Itr& operator++();
    bool operator==(const Itr rhs) const;
  };

  Itr begin() const;
  Itr end() const;
};
