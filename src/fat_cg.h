#include <armadillo>

using namespace arma;

/* conjugate gradient solver optimized
  for dealing with instances of FatGLM instances */
class FatCG {

public:
  void solve(const mat &x1, const mat &x2,
              const vec &b, vec &x,
              double multiplier, bool restart,
              size_t iterations=3);

private:
    const double RESIDUAL_TOL = 1e-3;
    double prev_r_sq_sum;
    arma::vec p_top, p_bottom, r_top, r_bottom;
};
