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

void TestGLM::solve(colvec &z, double lambda, size_t max_iterations){

  assert(lambda > 0);

  CG cg_solver;

  mat K_A;

  size_t i;

  uvec A, A_prev;

  colvec delz_A, g_A;

  const mat XX_I = XX + speye(XX.n_rows, XX.n_cols) * lambda * (1 - eta);
  const mat K = join_vert(join_horiz(XX, -XX), join_horiz(-XX, XX));
  const colvec g_init = g_start + lambda * eta;

  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, 2*n_half-1).unsafe_col(0);
  colvec w = u - l;

  for (i = 0; i < max_iterations; i++){

    const colvec g = g_init + K * z;

    findActiveSet(g, z, A);
    const size_t A_size = A.n_rows;

    if (A_size == 0) break;

    if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A_size){
      cg_solver.solve(K_A, g_A, delz_A, false, 3);
    }
    else {
      K_A = K(A, A);
      delz_A.zeros(A_size);
      g_A = -g(A);
      cg_solver.solve(K_A, g_A, delz_A, true, 3);
      A_prev = A;
    }

    if (norm(g_A, 2) <= 1) break;

    const colvec Kz = K_A * z(A);
    const colvec Ku = K_A * delz_A;
    if (! update(z, A, delz_A, Kz, Ku, g_init(A)))
      break;
    projectAndSparsify(w, u, l);
  }

  cout << "Iterations required: " << i << endl;
}