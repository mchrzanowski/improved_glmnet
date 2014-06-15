#include "fat_cg.h"
#include "fat_utils.h"

/* solve a special case where only 
  one of the matrices passed to the solve function
  is non-empty. */
void FatCG::subsolve(const mat &X, 
          const vec &b,
          vec &x,
          double multiplier,
          bool restart,
          size_t iterations){

  if (restart) {
    fatMultiply(X, x, multiplier, r_top);
    r_top -= b;
    p_top = -r_top;
    prev_r_sq_sum = dot(r_top, r_top);
  }

  for (size_t i = 0; i < iterations; i++){
    colvec Ap_top(p_top.n_rows);
    fatMultiply(X, p_top, multiplier, Ap_top);

    const double alpha = prev_r_sq_sum / dot(p_top, Ap_top);

    x += alpha * p_top;    
    r_top += alpha * Ap_top;
    
    const double r_sq_sum = dot(r_top, r_top);
    const double beta = r_sq_sum / prev_r_sq_sum;

    p_top *= beta;
    p_top -= r_top;
    prev_r_sq_sum = r_sq_sum;
  }
}

void FatCG::solve(const mat &x1, 
                  const mat &x2,
                  const vec &b,
                  vec &x,
                  double multiplier,
                  bool restart,
                  size_t iterations){

  const uword half = x1.n_cols;

  /* two special cases when one of x1 or x2 are empty
  that have to be special-cased, unfortunately. */
  if (x1.n_rows == 0 || x1.n_cols == 0){
    vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0);
    colvec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);
    subsolve(x2, b_bottom, x_bottom, multiplier, restart, iterations);
  }
  else if (x2.n_rows == 0 || x2.n_cols == 0){
    vec x_top = x.subvec(0, half-1).unsafe_col(0);
    const vec b_top = b.subvec(0, half-1).unsafe_col(0);
    subsolve(x1, b_top, x_top, multiplier, restart, iterations);
  }
  else {
    fullSolve(x1, x2, b, x, multiplier, restart, iterations);
  }
}

void FatCG::fullSolve(const mat &x1, 
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

  for (size_t i = 0; i < iterations; i++){
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
