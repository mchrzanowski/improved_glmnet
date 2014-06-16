#include "skinny_cg.h"
#include "skinny_utils.h"

/* solve a special case where only 
  one of the matrices passed to the solve function
  is non-empty. */
void SkinnyCG::subsolve(const mat &A, 
                        const vec &b,
                        vec &x,
                        bool restart,
                        size_t iterations){
  if (restart) {
    skinnyMultiply(A, x, r_top);
    r_top -= b;
    p_top = -r_top;
    prev_r_sq_sum = dot(r_top, r_top);
  }

  for (size_t i = 0; i < iterations; i++){
    colvec Ap_top(p_top.n_rows);
    skinnyMultiply(A, p_top, Ap_top);

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

/* perform conjugate gradient descent */
void SkinnyCG::solve(const mat &A1, const mat &A2, const mat &A4,
                      const vec &b,
                      vec &x,
                      bool restart,
                      size_t iterations){

  const uword half = A1.n_cols;
  /* two special cases when one of A1 or A4 are empty.*/
  if (A1.n_rows == 0 || A1.n_cols == 0){
    vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0); 
    colvec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);
    subsolve(A4, b_bottom, x_bottom, restart, iterations);
  }
  else if (A4.n_rows == 0 || A4.n_cols == 0){
    vec x_top = x.subvec(0, half-1).unsafe_col(0);
    const vec b_top = b.subvec(0, half-1).unsafe_col(0);
    subsolve(A1, b_top, x_top, restart, iterations);
  }
  else {
    fullSolve(A1, A2, A4, b, x, restart, iterations);
  }
}

void SkinnyCG::fullSolve(const mat &A1, const mat &A2, const mat &A4,
                          const vec &b,
                          vec &x,
                          bool restart,
                          size_t iterations){

  const uword half = A1.n_cols;

  /* we deal with halves, so split b and x up */
  vec x_top = x.subvec(0, half-1).unsafe_col(0);
  vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0);

  const vec b_top = b.subvec(0, half-1).unsafe_col(0);
  const vec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);

  if (restart) {
    skinnyMultiply(A1, A2, A4, x_top, x_bottom, r_top, r_bottom);
    r_top -= b_top;
    r_bottom -= b_bottom;
    p_top = -r_top;
    p_bottom = -r_bottom;
    prev_r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
  }

  for (size_t i = 0; i < iterations; i++){
    colvec Ap_top, Ap_bottom;
    skinnyMultiply(A1, A2, A4, p_top, p_bottom, Ap_top, Ap_bottom);

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