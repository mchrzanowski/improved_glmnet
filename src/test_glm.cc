#include <assert.h>
#include "cg.h"
#include "test_glm.h"
#include "utils.h"

using namespace arma;
using namespace std;

TestGLM::TestGLM(const mat &X, const vec &y, const double lambda,
    const double eta) : eta(eta), lambda(lambda) {

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);

    // construct Hessian matrix once and for all.
    const mat XXT = X.t();
    XX = XXT * X;
    mat XX_I = XX + speye(XX.n_rows, XX.n_cols) * lambda * (1 - eta);
    K = join_vert(join_horiz(XX_I, -XX), join_horiz(-XX, XX_I));

    const colvec Xy = XXT * y;
    g_start = join_vert(-Xy, Xy) + lambda * eta;
}

void TestGLM::solve(colvec &z, const size_t max_iterations){

    const size_t n_half = z.n_rows / 2;

    CG cg_solver;

    mat K_A;

    size_t i;

    uvec A, A_prev, D, neg_w(n_half), pos_w(n_half),
        neg_delz, nonpos_g(z.n_rows), pos_z(z.n_rows);

    colvec delz(z.n_rows), delz_A, g(g_start.n_rows), g_A;

    colvec u = z.subvec(0, n_half-1).unsafe_col(0);
    colvec l = z.subvec(n_half, 2*n_half-1).unsafe_col(0);
    colvec w = u - l;

    for (i = 0; i < max_iterations; i++){

        const colvec g = g_start + K * z;

        nonpos_g = find(g <= 0);
        pos_z = find(z > 0);
        vunion(nonpos_g, pos_z, A);
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
        update(z, A, delz_A, w, u, l, n_half);
    }

    cout << "Iterations required: " << i << endl;
}