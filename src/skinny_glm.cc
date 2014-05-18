#include "skinny_glm.h"
#include "skinny_cg.h"
#include "skinny_utils.h"
#include "utils.h"
#include <assert.h>

using namespace std;

SkinnyGLM::SkinnyGLM(const mat &X, const vec &y, const double eta) :
                      GLM(eta), n(2*X.n_cols), n_half(X.n_cols) {
  assert(eta >= 0 && eta <= 1);    
  XX = symmatu(X.t() * X);
  const colvec Xy = (y.t() * X).t();
  g_start = join_vert(-Xy, Xy);
}

void SkinnyGLM::createMatrixChunks(mat &x1, mat &x2, mat &x4,
                                    const uvec &A, const uvec &A_prev,
                                    double multiplier){
  
  // get the index of A that is at the boundary of the top
  // n elements and the bottom ones.
  uword divider = binarySearch(A, n_half);
  uvec A_top, A_bottom;
  cutVector(A_top, A_bottom, A, divider, n_half);

  // maybe I can get away with not needing to re-create
  // some of the matrices. this happens if A_top or A_bottom haven't changed
  // from the previous iteration.
  if (A_prev.n_rows > 0){
    uword A_prev_divider = binarySearch(A_prev, n_half);
    uvec A_prev_top, A_prev_bottom;
    cutVector(A_prev_top, A_prev_bottom, A_prev, A_prev_divider, n_half);

    bool changed = false;
    // too bad, the active set for the top n variables changed.
    if (A_top.n_rows != A_prev_top.n_rows 
        || accu(A_top == A_prev_top) != A_top.n_rows) {
      x1 = XX(A_top, A_top);
      x1.diag() += multiplier;
      changed = true;
    }

    // likewise for the bottom n variables.
    if (A_bottom.n_rows != A_prev_bottom.n_rows 
        || accu(A_bottom == A_prev_bottom) != A_bottom.n_rows) {
      x4 = XX(A_bottom, A_bottom);
      x4.diag() += multiplier;
      changed = true;
    }

    // if any change occurred in the active set,
    // we'll need to re-create x2.
    if (changed){
      x2 = -XX(A_top, A_bottom);
    }

  }
  // recreate all matrices.
  else {
    x1 = XX(A_top, A_top);
    x1.diag() += multiplier;
  
    x2 = -XX(A_top, A_bottom);
  
    x4 = XX(A_bottom, A_bottom);
    x4.diag() += multiplier;
  }

}

void SkinnyGLM::solve(colvec &z, double lambda, size_t max_iterations){

  assert(lambda > 0);
  const double multiplier = lambda * eta;

  SkinnyCG cg_solver;
  colvec delz_A, g_A, g_start_with_multi_A;
  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;
  mat x1, x2, x4;
  size_t i;
  uvec A, A_prev;

  const colvec g_start_with_multi = g_start + multiplier;

  for (i = 0; i < max_iterations; i++){

    const colvec g_half = XX * w;
    colvec g = g_start_with_multi;
    g.subvec(0, n_half-1) += g_half + u * multiplier;
    g.subvec(n_half, n-1) += -g_half + l * multiplier;

    findActiveSet(g, z, A);
    if (A.n_rows == 0) break;

    // if the active set hasn't changed since the prev iteration,
    // then we can take a few more CG steps in this direction
    // and don't have to re-create the matrix chunks.
    if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
        cg_solver.solve(x1, x2, x4, g_A, delz_A, false);
    }
    else {
      createMatrixChunks(x1, x2, x4, A, A_prev, multiplier);
      delz_A.zeros(A.n_rows);
      g_A = -g(A);
      g_start_with_multi_A = g_start_with_multi(A);
      cg_solver.solve(x1, x2, x4, g_A, delz_A, true);
      A_prev = A;
    }
    
    if (norm(g_A, 2) <= G_A_TOL) break;

    const uword divider = x1.n_cols;

    // we need K * z. but we actually calculated
    // that as part of the gradient.
    const colvec K_z_A = -g_A - g_start_with_multi_A;

    colvec delz_A_top, delz_A_bottom;
    cutVector(delz_A_top, delz_A_bottom, delz_A, divider, 0);

    colvec K_u_A(A.n_rows);
    skinnyMultiply(x1, x2, x4, delz_A_top, delz_A_bottom, K_u_A);

    if (! update(z, A, delz_A, K_z_A, K_u_A, g_start_with_multi_A))
      break;
    projectAndSparsify(w, u, l);
  }
  //cout << "Iterations required: " << i << endl;
}
