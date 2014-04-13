#include "fat_glm.h"
#include "cg.h"
#include "utils.h"
#include <assert.h>

using namespace arma;
using namespace std;

FatGLM::FatGLM(const mat &X_, const vec &y, const double lambda, const double eta) :
X(X_), multiplier(lambda * (1 - eta)) {

    cout << &X << "\t" << &X_ << endl;

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);

    m = X.n_rows;
    n_half = X.n_cols;
    n = 2*n_half;

    XT = X.t();
    const colvec Xy = XT * y;
    g_start.zeros(n);
    g_start.subvec(0, n_half-1) = -Xy + lambda * eta;
    g_start.subvec(n_half, n-1) = Xy + lambda * eta;
}

void FatGLM::createMatrixChunks(mat &x1_pre, mat &x1_post, mat &x2_pre, 
    mat &x2_post, mat &x4_pre, mat &x4_post, const uvec &A,
    const size_t n_half, uword &divider){
    
    const uvec top = find(A < n_half);
    divider = top(top.n_rows-1) + 1;

    const uvec A_top = static_cast<uvec>(A(top));
    const uvec A_bottom = static_cast<uvec>(A.subvec(divider,
        A.n_rows-1) - n_half);

    x1_pre = X.cols(A_top);
    x1_post = XT.rows(A_top);
    
    x2_pre = X.cols(A_bottom);
    x2_post = -XT.rows(A_top);
    
    x4_pre = X.cols(A_bottom);
    x4_post = XT.rows(A_bottom);
}

void FatGLM::solve(colvec &z, const size_t max_iterations){

    CG cg_solver;

    size_t i;

    uvec A, A_prev, D, neg_w(n_half), pos_w(n_half),
        neg_delz, nonpos_g(z.n_rows), pos_z(z.n_rows);

    colvec delz(z.n_rows), delz_A, g(g_start.n_rows), g_A;

    const colvec u = z.subvec(0, n_half-1).unsafe_col(0);
    const colvec l = z.subvec(n_half, n-1).unsafe_col(0);
    colvec w = u - l;

    mat x1_pre, x1_post, x2_pre, x2_post, x4_pre, x4_post;
    uword divider = 0;

    for (i = 0; i < max_iterations; i++){

        const colvec g_half = XT * (X * w);

        g = g_start;
        g.subvec(0, n_half-1) += g_half + u * multiplier;
        g.subvec(n_half, n-1) += -g_half + l * multiplier;

        nonpos_g = find(g <= 0);
        pos_z = find(z > 0);
        vunion(nonpos_g, pos_z, A);

        if (A.n_rows == 0) break;

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
            cg_solver.solve(x1_pre, x1_post, x2_pre, x2_post,
                x4_pre, x4_post, g_A, delz_A, divider, true, 3);
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
            createMatrixChunks(x1_pre, x1_post, x2_pre,
                x2_post, x4_pre, x4_post, A, n_half, divider);
            delz_A.zeros(A.n_rows);
            g_A = -g(A);
            
            cg_solver.solve(x1_pre, x1_post, x2_pre, x2_post,
                x4_pre, x4_post, g_A, delz_A, divider, true, 3);
            
            A_prev = A;
        }
        
        if (norm(g_A, 2) <= 1) break;

        delz.zeros();
        delz(A) = delz_A;

        // select step size...
        neg_delz = A(find(delz_A < 0));
        vintersection(neg_delz, pos_z, D);
        if (D.n_rows == 0) break;

        const vec alphas = z(D) / delz(D);
        double alpha = min(-max(alphas), 1.0);
        assert(alpha > 0);

        z(A) += delz_A * alpha;
        z.transform([] (double val) { return max(val, 0.); });

        // force one of indices of z to be active....
        w = u - l; 
        neg_w = find(w < 0);
        pos_w = find(w > 0);
        z(neg_w).zeros();
        z(pos_w + n_half).zeros();
        z(neg_w + n_half) = -w(neg_w);
        z(pos_w) = w(pos_w);
    }

    cout << "Iterations required: " << i << endl;
}
