#include "skinny_glm.h"
#include "cg.h"
#include "utils.h"
#include <assert.h>

using namespace arma;
using namespace std;

SkinnyGLM::SkinnyGLM(const mat &X, const vec &y, const double lambda, const double eta) :
multiplier(lambda * (1 - eta)) {

    m = X.n_rows;
    n_half = X.n_cols;
    n = 2*n_half;

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);
    
    XX = X.t() * X;
    const colvec Xy = (y.t() * X).t();
    g_start.zeros(n);
    g_start.subvec(0, n_half-1) = -Xy + lambda * eta;
    g_start.subvec(n_half, n-1) = Xy + lambda * eta;
}

void SkinnyGLM::createMatrixChunks(mat &x1, mat &x2, mat &x4, const uvec &A,
const size_t n_half, uword &divider){
    const uvec top = find(A < n_half);
    divider = top(top.n_rows-1) + 1;

    const uvec A_top = static_cast<uvec>(A(top));
    const uvec A_bottom = static_cast<uvec>(A.subvec(divider,
        A.n_rows-1) - n_half);

    x1 = XX(A_top, A_top);
    x1.diag() += multiplier;
    
    x2 = -XX(A_top, A_bottom);
    
    x4 = XX(A_bottom, A_bottom);
    x4.diag() += multiplier;
}

void SkinnyGLM::solve(colvec &z, const size_t max_iterations){

    CG cg_solver;
    colvec delz_A, g_A;
    const colvec u = z.subvec(0, n_half-1).unsafe_col(0);
    const colvec l = z.subvec(n_half, n-1).unsafe_col(0);
    colvec w = u - l;
    mat x1, x2, x4;
    size_t i;
    uvec A, A_prev, D;
    uword divider = 0;

    for (i = 0; i < max_iterations; i++){

        const colvec g_half = XX * w;
        colvec g = g_start;
        g.subvec(0, n_half-1) += g_half + u * multiplier;
        g.subvec(n_half, n-1) += -g_half + l * multiplier;

        const uvec nonpos_g = find(g <= 0);
        const uvec pos_z = find(z > 0);
        vunion(nonpos_g, pos_z, A);

        if (A.n_rows == 0) break;

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
            cg_solver.solve(x1, x2, x4, g_A, delz_A, divider, true, 3);
        }
        /*else if (lol.n_rows == A.n_rows) {

            uvec diff;
            vdifference(A_prev, lol, diff);

            int A_index = A_prev.n_rows - 1;
            for (int s = diff.n_rows - 1; s >= 0 && A_index >= 0; s--){
                while (diff(s) != A_prev(A_index)) A_index--;
                K_A.shed_row(s);
                K_A.shed_col(s);
                delz_A.shed_row(s);
                g_A.shed_row(s);
                cg_solver.getP().shed_row(s);
                cg_solver.getR().shed_row(s);
            }

            cg_solver.solve(K_A, g_A, delz_A, false, 3);
        }*/
        else {
            createMatrixChunks(x1, x2, x4, A, n_half, divider);
            delz_A.zeros(A.n_rows);
            g_A = -g(A);
            cg_solver.solve(x1, x2, x4, g_A, delz_A, divider, true, 3);
            A_prev = A;
        }
        
        if (norm(g_A, 2) <= 1) break;

        colvec delz = zeros<vec>(n);
        delz(A) = delz_A;

        // select step size...
        const uvec neg_delz = A(find(delz_A < 0));
        vintersection(neg_delz, pos_z, D);
        if (D.n_rows == 0) break;

        const vec alphas = z(D) / delz(D);
        double alpha = min(-max(alphas), 1.0);
        assert(alpha > 0);

        z(A) += delz_A * alpha;
        z.transform([] (double val) { return max(val, 0.); });

        // force one of indices of z to be active....
        w = u - l; 
        const uvec neg_w = find(w < 0);
        const uvec pos_w = find(w > 0);
        z(neg_w).zeros();
        z(pos_w + n_half).zeros();
        z(neg_w + n_half) = -w(neg_w);
        z(pos_w) = w(pos_w);
    }

    cout << "Iterations required: " << i << endl;
}
