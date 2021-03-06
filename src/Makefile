# Makefile based *HEAVILY* on Jerry Cain's CS110 Makefiles

CXX = g++
RM = rm -f
OpenBLAS_DIR = /opt/OpenBLAS

# flags
CXXFLAGS = -std=c++11 -march=native -Wall -O4 -I$(OpenBLAS_DIR)/include \
						-L$(OpenBLAS_DIR)/lib
LDFLAGS = -lopenblas -larmadillo

# In this section, you list the files that are part of the project.
# If you add/change names of header/source files, here is where you
# edit the Makefile.
SOURCES = cg.cc glm.cc skinny_glm.cc fat_glm.cc test_glm.cc \
	fat_utils.cc fat_cg.cc skinny_cg.cc skinny_utils.cc glm_factory.cc
HEADERS = $(SOURCES:.cc=.h) utils.h
OBJECTS = $(SOURCES:.cc=.o)

CVX = cvx_validation.out
CV = cross_validation.out
RP = regularization_path.out

all: examples
cv: $(CV)
cvx: $(CVX)
examples: $(CVX) $(CV) $(RP)
rp: $(RP)

cross_validation.out: $(OBJECTS) cross_validation.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cvx_validation.out: $(OBJECTS) cvx_validation.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

regularization_path.out: $(OBJECTS) regularization_path.cc
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
.PHONY: clean spartan cv cvx rp all examples

clean:
	$(RM) $(CVX) $(CV) $(RP) $(OBJECTS) Makefile.dependencies

spartan: clean
	$(RM) *~

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
