#include "glm.h"

/* A class that isn't optimized for speed but
  rather for readability and pedagogy.
  Used to test things out before implementing
  them in the optimized solvers */
class TestGLM : public GLM {

public:
  TestGLM(const arma::mat &X, const arma::vec &y, const double eta);
  
  void solve(arma::colvec &z, const double lambda, 
              const size_t max_iterations);

private:
  arma::mat XX, K;
  const arma::uword n_half;
};