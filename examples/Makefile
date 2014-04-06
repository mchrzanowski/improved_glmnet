# tools
CXX := g++
RM := rm -f

# flags
CCFLAGS = -std=c++0x -Wall -O3
INCLUDES = -I/usr/include -I/usr/local/include
LIBS = -L/usr/local/lib -L/opt/intel/lib/intel64 -L/opt/intel/mkl/lib/intel64
LDFLAGS = -larmadillo -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

# sources
sources := basic_armadillo.cc
targets := $(sources:.cc=.out)

# gmake magic
.PHONY: default all clean

#targets
default: all
all:
	$(CXX) $(CCFLAGS) $(INCLUDES) $(LIBS) -o basic_armadillo.out basic_armadillo.cc $(LDFLAGS)

clean:
	$(RM) $(targets)
