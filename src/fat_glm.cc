#include <assert.h>
#include "cg.h"
#include "fat_glm.h"
#include "utils.h"

using namespace arma;
using namespace std;

FatGLM::FatGLM(const mat &X_, const vec &y, double eta) :
                X(X_), n(2*X.n_cols), n_half(X.n_cols), eta(eta) {

  assert(eta >= 0 && eta <= 1);
  const colvec Xy = (y.t() * X).t();
  g_start.zeros(n);
  g_start.subvec(0, n_half-1) = -Xy;
  g_start.subvec(n_half, n-1) = Xy;
}

void FatGLM::createMatrixChunks(mat &x1, mat &x2,
                                const uvec &A, const uvec &A_prev){
  
  uword divider = binarySearch(A, n_half);
  
  const uvec A_top = A.subvec(0, divider-1);
  const uvec A_bottom = A.subvec(divider, A.n_rows-1) - n_half;

  // maybe I can get away with not needing to re-create
  // x1 or x2. this happens if A_top or A_bottom haven't changed
  // from the previous iteration.
  if (A_prev.n_rows > 0){
    uword A_prev_divider = binarySearch(A_prev, n_half);
    const uvec A_prev_top = A_prev.subvec(0, A_prev_divider-1);
    const uvec A_prev_bottom = A_prev.subvec(A_prev_divider, 
                                              A_prev.n_rows-1) - n_half;

    if (A_top.n_rows != A_prev_top.n_rows 
        || accu(A_top == A_prev_top) != A_top.n_rows) {
      x1 = X.cols(A_top);
    }

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

void FatGLM::solve(colvec &z, double lambda, size_t max_iterations){

  assert(lambda > 0);
  const double multiplier = lambda * eta;
  const colvec g_init = g_start + multiplier;

  CG cg_solver;
  colvec delz_A, g_A, g_init_A;
  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;
  mat x1, x2;
  size_t i;
  uvec A, A_prev;

  for (i = 0; i < max_iterations; i++){

    const colvec g_half = ((X * w).t() * X).t();
    colvec g = g_init;
    g.subvec(0, n_half-1) += g_half + u * multiplier;
    g.subvec(n_half, n-1) += -g_half + l * multiplier;
    
    const uvec nonpos_g = find(g <= 0);
    const uvec pos_z = find(z > 0);
    vunion(nonpos_g, pos_z, A);

    if (A.n_rows == 0) break;

    if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
      cg_solver.fatMatrixSolve(x1, x2, g_A,
                                delz_A, multiplier, true, 3);
    }
    else {
      createMatrixChunks(x1, x2, A, A_prev);
      delz_A.zeros(A.n_rows);
      g_A = -g(A);
      g_init_A = g_init(A);
      cg_solver.fatMatrixSolve(x1, x2, g_A,
                                delz_A, multiplier, true, 3);
      A_prev = A;
    }

    if (norm(g_A, 2) <= .5) break;
    const uword divider = x1.n_cols;

    const colvec z_A_top = u(A.subvec(0, divider - 1));
    const colvec z_A_bottom = l(A.subvec(divider, A.n_rows-1) - n_half);

    colvec Kz(A.n_rows);
    fatMultiply(x1, x2, z_A_top, z_A_bottom, multiplier, Kz);

    const colvec delz_A_top = delz_A.subvec(0, divider - 1);
    const colvec delz_A_bottom = delz_A.subvec(divider, delz_A.n_rows-1);

    colvec Ku(A.n_rows);
    fatMultiply(x1, x2, delz_A_top, delz_A_bottom, multiplier, Ku);

    bool progress_made = updateBetter(z, A, delz_A, Kz, Ku, g_init_A);
    if (! progress_made) break;
    projectAndSparsify(w, u, l);

  }
  cout << "Iterations required: " << i << endl;
}
