GCC_FLAGS=-g -std=gnu++20 -Wall -Wno-deprecated -Wdeprecated-declarations -Wno-error=deprecated-declarations -Wno-sign-compare -Wno-unused -Wunused-label -Wunused-result -Wnon-virtual-dtor -fopenmp
CPP_LIBS=-lfolly -lglog -levent -lgflags -lpthread -lgtest -lfmt

SRCS_CC := $(wildcard *.cc)
SRCS_CPP := $(wildcard *.cpp)
BINS := $(SRCS_CC:.cc=.bin) $(SRCS_CPP:.cpp=.bin)

all: $(BINS)

%.bin: %.cc
	g++ $< -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

%.bin: %.cpp
	g++ $< -O3 $(GCC_FLAGS) $(CPP_LIBS) -o $@

clean:
	rm -f *.bin

