#include "glm.h"

using namespace arma;

/* A class that isn't optimized for speed but
  rather for readability and pedagogy.
  Used to test things out before implementing
  them in the optimized solvers */
class TestGLM : public GLM {

public:
  TestGLM(const mat &X, const vec &y, const double eta);
  
  size_t solve(colvec &z,
              double lambda,
              size_t max_iterations=0);
  
  size_t sequential_solve(colvec &z,
                          double lambda, double prev_lambda,
                          size_t max_iterations=0);

  size_t solve(colvec &z, colvec &g,
                double lambda,
                const uvec *blacklisted=NULL,
                size_t max_iterations=0);

private:
  mat XX, K;
  const uword n_half, n;
};