GCC_FLAGS=-g -std=gnu++20 -Wall -Wno-deprecated -Wdeprecated-declarations -Wno-error=deprecated-declarations -Wno-sign-compare -Wno-unused -Wunused-label -Wunused-result -Wnon-virtual-dtor -fopenmp -DNDEBUG
CPP_LIBS=-lglog -levent -lgflags -lpthread -lgtest -lfmt

SRCS_CC := $(filter-out prime_number_gen.cc prime_number_gen_test.cc, $(wildcard *.cc))
SRCS_CPP := $(wildcard *.cpp)
BINS := $(SRCS_CC:.cc=.bin) $(SRCS_CPP:.cpp=.bin) prime_number_gen_test.bin

all: $(BINS)

# Special rule for 2022-03.bin: this binary requires both 2022-03.cc and prime_number_gen.cc to be compiled and linked together.
# The generic pattern rule does not handle this dependency, so we specify it explicitly here.
2022-03.bin: 2022-03.cc prime_number_gen.cc
	g++ $^ -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

# Specific rule for 2025-12.bin which depends on prime_number_gen.cc
2025-12.bin: 2025-12.cc prime_number_gen.cc
	g++ $^ -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

# Specific rule for prime_number_gen_test.bin
prime_number_gen_test.bin: prime_number_gen_test.cc prime_number_gen.cc
	g++ $^ -O3 $(GCC_FLAGS) -lglog -lgflags -lpthread -lgtest -lfmt -lbenchmark -o $@

%.bin: %.cc
	g++ $< -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

%.bin: %.cpp
	g++ $< -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

clean:
	rm -f *.bin

