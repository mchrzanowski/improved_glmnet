#include "fat_cg.h"
#include "fat_utils.h"

void FatCG::solve(const mat &x1, 
                  const mat &x2,
                  const vec &b,
                  vec &x,
                  double multiplier,
                  bool restart,
                  size_t iterations){

  const uword half = x1.n_cols;

  vec x_top = x.subvec(0, half-1).unsafe_col(0);
  vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0);

  const vec b_top = b.subvec(0, half-1).unsafe_col(0);
  const vec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);

  if (restart) {

    fatMultiply(x1, x2, x_top, x_bottom, multiplier, r_top, r_bottom);
    
    r_top -= b_top;
    r_bottom -= b_bottom;

    p_top = -r_top;
    p_bottom = -r_bottom;
    prev_r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
  }

  for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
    colvec Ap_top(p_top.n_rows), Ap_bottom(p_bottom.n_rows);
    fatMultiply(x1, x2, p_top, p_bottom, multiplier, Ap_top, Ap_bottom);

    const double alpha = prev_r_sq_sum / 
        (dot(p_top, Ap_top) + dot(p_bottom, Ap_bottom));

    x_top += alpha * p_top;
    x_bottom += alpha * p_bottom;
    
    r_top += alpha * Ap_top;
    r_bottom += alpha * Ap_bottom;
    
    const double r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
    const double beta = r_sq_sum / prev_r_sq_sum;

    p_top *= beta;
    p_top -= r_top;
    
    p_bottom *= beta;
    p_bottom -= r_bottom;
    
    prev_r_sq_sum = r_sq_sum;
  }
}