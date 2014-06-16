#include <armadillo>

using namespace arma;

/* conjugate gradient solver optimized
  for dealing with instances of SkinnyGLM instances */
class SkinnyCG {

public:
  void solve(const mat &A1, 
              const mat &A2, const mat &A4,
              const vec &b,
              vec &x,
              bool restart,
              size_t iterations=3);
private:
  void subsolve(const mat &A,
                const vec &b,
                vec &x,
                bool restart,
                size_t iterations);

  void fullSolve(const mat &A1, const mat &A2, const mat &A4,
                  const vec &b,
                  vec &x,
                  bool restart,
                  size_t iterations);

  double prev_r_sq_sum;
  vec p_top, p_bottom, r_top, r_bottom;  
};
