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

  vec x_top;
  vec b_top;
  if (half > 0){
    x_top = x.subvec(0, half-1);
    b_top = b.subvec(0, half-1);
  }
  
  vec x_bottom;
  vec b_bottom;
  if (half < x.n_rows){
    x_bottom = x.subvec(half, x.n_rows-1);
    b_bottom = b.subvec(half, x.n_rows-1);
  }

  if (restart) {
    fatMultiply(x1, x2, x_top, x_bottom, multiplier, r_top, r_bottom);

    if (x_top.n_rows > 0){
      r_top -= b_top;
      p_top = -r_top;
    }
    if (x_bottom.n_rows > 0){
      r_bottom -= b_bottom;
      p_bottom = -r_bottom;
    }
    prev_r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
  }

  for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
    colvec Ap_top(p_top.n_rows), Ap_bottom(p_bottom.n_rows);
    fatMultiply(x1, x2, p_top, p_bottom, multiplier, Ap_top, Ap_bottom);

    const double alpha = prev_r_sq_sum / 
        (dot(p_top, Ap_top) + dot(p_bottom, Ap_bottom));

    if (x_top.n_rows > 0){
      x_top += alpha * p_top;
      r_top += alpha * Ap_top;
    }
    if (x_bottom.n_rows > 0){
      x_bottom += alpha * p_bottom;
      r_bottom += alpha * Ap_bottom;
    }
   
    const double r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
    const double beta = r_sq_sum / prev_r_sq_sum;

    if (p_top.n_rows > 0){
      p_top *= beta;
      p_top -= r_top;
    }
    
    if (p_bottom.n_rows > 0){
     p_bottom *= beta;
      p_bottom -= r_bottom; 
    }
    prev_r_sq_sum = r_sq_sum;
  }

  if (half > 0){
    x.subvec(0, half-1) = x_top;
  }
  
  if (half < x.n_rows){
    x.subvec(half, x.n_rows-1) = x_bottom;
  }

}