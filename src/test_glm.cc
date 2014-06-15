#include <assert.h>
#include "cg.h"
#include "test_glm.h"
#include "utils.h"

using namespace arma;
using namespace std;

TestGLM::TestGLM(const mat &X, const vec &y, double eta) : 
                  GLM(eta), n_half(X.n_cols) {

  assert(eta >= 0 && eta <= 1);

  const mat XXT = X.t();
  XX = XXT * X;

  const colvec Xy = XXT * y;
  g_start = join_vert(-Xy, Xy);
}

size_t TestGLM::sequential_solve(colvec &z,
                                double lambda, double prev_lambda,
                                size_t max_iterations){
  
  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;

  const colvec yX = g_start.subvec(n_half, 2*n_half-1);

  colvec val = abs(yX - XX * w);
  uvec excluded = find(val < eta * (2 * lambda - prev_lambda));
  excluded = join_vert(excluded, excluded + n_half);

  colvec g;
  uvec active, to_unexclude;

  size_t iters = 0;

  while (true){

    iters += solve(z, g, lambda, &excluded, max_iterations);

    /* check KKT conditions.
    is an excluded variable in the active set?
    if so, we screwed up. remove it from the blacklist
    and re-do the optimization. */
    findActiveSet(g, z, active);
    vintersection(excluded, active, to_unexclude);
    if (to_unexclude.n_rows > 0){
      vdifference(excluded, to_unexclude, excluded);
    }
    else {
      break;
    }
  }
  return iters;
}

size_t TestGLM::solve(colvec &z,
                    double lambda,
                    size_t max_iterations){
  colvec g;
  return solve(z, g, lambda, NULL, max_iterations);
}

size_t TestGLM::solve(colvec &z, colvec &g,
                      double lambda,
                      const uvec *blacklisted,
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
  colvec l = z.subvec(n_half, 2*n_half-1).unsafe_col(0);
  colvec w = u - l;

  for (i = 0; i < max_iterations; i++){

    g = g_bias + K * z;

    findActiveSet(g, z, A);
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
