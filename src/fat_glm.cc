#include <assert.h>
#include "fat_cg.h"
#include "fat_glm.h"
#include "fat_utils.h"
#include "cg.h"
#include "utils.h"

using namespace arma;

FatGLM::FatGLM(const mat &X, const vec &y, double eta) :
                GLM(eta, X.n_cols, 2*X.n_cols), X(X) {
  const colvec Xy = (y.t() * X).t();
  g_start = join_vert(-Xy, Xy);
}

/* calculate ret = X^T X * w */
void
FatGLM::calculateXXw(const colvec &w, colvec &ret){
  ret = ((X * w).t() * X).t();
}

/* x1 and x2 are the rows from X based on the active set A.
x1 comes from the first n vals of X, x2 the bottom. */
void
FatGLM::createMatrixChunks(mat &x1, mat &x2,
                            const uvec &A, const uvec &A_prev){
  
  // get the index of A that is at the boundary of the top
  // n elements and the bottom ones.
  uword divider = binarySearch(A, n_half);
  uvec A_top, A_bottom;
  cutVector(A_top, A_bottom, A, divider, n_half);

  // maybe I can get away with not needing to re-create
  // x1 or x2. this happens if A_top or A_bottom haven't changed
  // from the previous iteration.
  if (A_prev.n_rows > 0){
    
    uword A_prev_divider = binarySearch(A_prev, n_half);
    uvec A_prev_top, A_prev_bottom;
    cutVector(A_prev_top, A_prev_bottom, A_prev, A_prev_divider, n_half);

    // too bad, the active set for the top n variables changed.
    if (A_top.n_rows != A_prev_top.n_rows 
        || accu(A_top == A_prev_top) != A_top.n_rows) {
      x1 = X.cols(A_top);
    }

    // likewise for the bottom n variables.
    if (A_bottom.n_rows != A_prev_bottom.n_rows 
        || accu(A_bottom == A_prev_bottom) != A_bottom.n_rows) {
      x2 = X.cols(A_bottom);
    }
  }
  else {
    x1 = X.cols(A_top);
    x2 = X.cols(A_bottom);
  }
}

size_t
FatGLM::solve(colvec &z, colvec &g,
              double lambda,
              const uvec *whitelisted,
              size_t max_iterations){

  assert(lambda > 0);
  if (max_iterations == 0){
    max_iterations = z.n_rows;
  }

  FatCG cg_solver;
  colvec delz_A, g_A, g_bias_A, Kz_A;
  mat x1, x2;
  size_t i;
  uvec A, A_prev;

  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;
  
  const colvec g_bias = g_start + lambda * eta;
  const double multiplier = lambda * (1 - eta);

  for (i = 0; i < max_iterations; i++){

    /* use the gradient from the sequential_solve's KKT check
      for the first iteration of this run. */
    if (i != 0 || g.n_rows != z.n_rows){
      calculateGradient(z, lambda, g);
    }

    findActiveSet(g, z, A);
    if (whitelisted != NULL){
      vintersection(A, *whitelisted, A);
    }
    if (A.n_rows == 0) break;
    
    // if the active set hasn't changed since the prev iteration,
    // then we can take a few more CG steps in this direction
    // and don't have to re-create the matrix chunks.
    if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
      cg_solver.solve(x1, x2, g_A, delz_A, multiplier, false, 5 + (i / 100));
    }
    else {
      createMatrixChunks(x1, x2, A, A_prev);
      delz_A.zeros(A.n_rows);
      g_A = -g(A);
      g_bias_A = g_bias(A);
      cg_solver.solve(x1, x2, g_A, delz_A, multiplier, true, 5 + (i / 100));
      A_prev = A;
    }
    if (norm(g_A, 2) <= G_A_TOL) break;

    // we need K * z. but we actually calculated
    // that as part of the gradient.
    const colvec Kz_A = -g_A - g_bias_A;

    const uword divider = x1.n_cols;
    colvec delz_A_top, delz_A_bottom;
    cutVector(delz_A_top, delz_A_bottom, delz_A, divider, 0);

    colvec Ku_A(A.n_rows);
    fatMultiply(x1, x2, delz_A_top, delz_A_bottom, multiplier, Ku_A);

    if (! update(z, A, delz_A, Kz_A, Ku_A, g_bias_A)) break;

    projectAndSparsify(w, u, l);
  }
  return i;
}
