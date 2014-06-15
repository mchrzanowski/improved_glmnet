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
  void fullSolve(const mat &x1, 
                  const mat &x2,
                  const vec &b,
                  vec &x,
                  double multiplier,
                  bool restart,
                  size_t iterations);

  void subsolve(const mat &X, 
                const vec &b,
                vec &x,
                double multiplier,
                bool restart,
                size_t iterations);

  double prev_r_sq_sum;
  vec p_top, p_bottom, r_top, r_bottom;
};
