#include <assert.h>
#include "cg.h"
#include "test_glm.h"
#include "utils.h"

using namespace arma;

TestGLM::TestGLM(const mat &X, const vec &y, double eta) : 
                  GLM(eta, X.n_cols, 2*X.n_cols) {
  const mat XXT = X.t();
  XX = XXT * X;
  const colvec Xy = XXT * y;
  g_start = join_vert(-Xy, Xy);
}

/* calculate ret = X^T X * w */
void
TestGLM::calculateXXw(const colvec &w, colvec &ret){
  ret = XX * w;
}

size_t
TestGLM::solve(colvec &z, colvec &g,
                double lambda,
                const uvec *whitelisted,
                size_t max_iterations){

  assert(lambda > 0);
  if (max_iterations == 0){
    max_iterations = z.n_rows;
  }

  CG cg_solver;
  mat K_A;
  size_t i;
  uvec A, A_prev;
  colvec delz_A, g_A;

  const mat XX_I = XX + speye(XX.n_rows, XX.n_cols) * lambda * (1 - eta);
  const mat K = join_vert(join_horiz(XX_I, -XX), join_horiz(-XX, XX_I));
  const colvec g_bias = g_start + lambda * eta;

  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;

  for (i = 0; i < max_iterations; i++){

    g = g_bias + K * z;

    findActiveSet(g, z, A);
    if (whitelisted != NULL){
      vintersection(A, *whitelisted, A);
    }
    if (A.n_rows == 0) break;

    if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
      cg_solver.solve(K_A, g_A, delz_A, false, 3);
    }
    else {
      K_A = K(A, A);
      delz_A.zeros(A.n_rows);
      g_A = -g(A);
      cg_solver.solve(K_A, g_A, delz_A, true, 3);
      A_prev = A;
    }
    if (norm(g_A, 2) <= G_A_TOL) break;

    const colvec Kz_A = K_A * z(A);
    const colvec Ku_A = K_A * delz_A;

    if (! update(z, A, delz_A, Kz_A, Ku_A, g_bias(A))) break;
    projectAndSparsify(w, u, l);
  }
  return i;
}
