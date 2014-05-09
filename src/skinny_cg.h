#include <armadillo>

using namespace arma;

class SkinnyCG {

public:
  void solve(const mat &x1, 
              const mat &x2, const mat &x4,
              const vec &b, vec &x,
              bool restart,
              size_t iterations=3);

private:
    const double RESIDUAL_TOL = 1e-3;
    double prev_r_sq_sum;
    vec p_top, p_bottom, r_top, r_bottom;  
};
