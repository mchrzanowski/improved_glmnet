# Makefile based *HEAVILY* on Jerry Cain's CS110 Makefiles

CXX = g++
RM = rm -f

# flags
CXXFLAGS = -std=c++0x -march=native -Wall -O4 -I/usr/include -I/usr/local/include \
	-L/usr/local/lib -L/opt/intel/lib/intel64 -L/opt/intel/mkl/lib/intel64
LDFLAGS = -larmadillo -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

# In this section, you list the files that are part of the project.
# If you add/change names of header/source files, here is where you
# edit the Makefile.
SOURCES = cg.cc utils.cc glm.cc skinny_glm.cc fat_glm.cc test_glm.cc \
	fat_utils.cc fat_cg.cc skinny_cg.cc
HEADERS = $(SOURCES:.cc=.h)
OBJECTS = $(SOURCES:.cc=.o)
TARGETS = glmnet.out
TESTS = matlab_validation.out

default: $(TARGETS)

test: $(TESTS)

glmnet.out: $(OBJECTS) main.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

matlab_validation.out: $(OBJECTS) matlab_validation.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# In make's default rules, a .o automatically depends on its .cc file
# (so editing the .cc will cause recompilation into its .o file).
# The line below creates additional dependencies, most notably that it
# will cause the .cc to rocompiled if any included .h file changes.

Makefile.dependencies:: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.dependencies

-include Makefile.dependencies

# Phony means not a "real" target, it doesn't build anything
# The phony target "clean" is used to remove all compiled object files.
# The phony target "spartan" is used to remove all compiled object and backup files.
.PHONY: clean spartan

clean:
	$(RM) $(TARGETS) $(OBJECTS) $(TESTS) Makefile.dependencies

spartan: clean
	$(RM) *~
