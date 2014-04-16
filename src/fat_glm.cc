#include <assert.h>
#include "cg.h"
#include "fat_glm.h"
#include "utils.h"

using namespace arma;
using namespace std;

FatGLM::FatGLM(const mat &X_, const vec &y, const double lambda,
    const double eta) :X(X_), multiplier(lambda * (1 - eta)),
    m(X.n_rows), n(2*X.n_cols), n_half(X.n_cols) {

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);

    const colvec Xy = (y.t() * X).t();
    g_start.zeros(n);
    g_start.subvec(0, n_half-1) = -Xy + lambda * eta;
    g_start.subvec(n_half, n-1) = Xy + lambda * eta;
}

void FatGLM::createMatrixChunks(mat &x1, mat &x2_pre,
    const uvec &A, const size_t n_half, uword &divider){
    
    const uvec top = find(A < n_half);
    divider = top(top.n_rows-1) + 1;

    const uvec A_top = static_cast<uvec>(A(top));
    const uvec A_bottom = static_cast<uvec>(A.subvec(divider,
        A.n_rows-1) - n_half);

    x1 = X.cols(A_top);
    
    x2_pre = X.cols(A_bottom);

}

void FatGLM::solve(colvec &z, const size_t max_iterations){

    CG cg_solver;
    colvec delz_A, g_A;
    const colvec u = z.subvec(0, n_half-1).unsafe_col(0);
    const colvec l = z.subvec(n_half, n-1).unsafe_col(0);
    colvec w = u - l;
    mat x1, x2_pre;
    size_t i;
    uvec A, A_prev, D;
    uword divider = 0;

    for (i = 0; i < max_iterations; i++){

        const colvec g_half = ((X * w).t() * X).t();
        colvec g = g_start;
        g.subvec(0, n_half-1) += g_half + u * multiplier;
        g.subvec(n_half, n-1) += -g_half + l * multiplier;

        const uvec nonpos_g = find(g <= 0);
        const uvec pos_z = find(z > 0);
        vunion(nonpos_g, pos_z, A);

        if (A.n_rows == 0) break;

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
            cg_solver.solve(x1, x2_pre, g_A,
                delz_A, divider, multiplier, true, 3);
        }
        /*else if (intersect.n_rows == A.n_rows && ((double) intersect.n_rows / A_prev.n_rows) > .75) {
            cout << "HERE" << endl;
            uvec diff;
            vdifference(A_prev, intersect, diff);

            int A_index = A_prev.n_rows - 1;
            for (int s = diff.n_rows - 1; s >= 0 && A_index >= 0; s--){
                while (diff(s) != A_prev(A_index)) A_index--;
                if (A_index < divider){
                    x1.shed_col(s);
                    x2_post.shed_row(s);
                }
                else {
                    x2_pre.shed_col(s - divider);
                }
                delz_A.shed_row(s);
                g_A.shed_row(s);
            }
            cout << "DONE" << endl;
            printf("%ux%u\n", x1.n_rows, x1.n_cols);
            printf("%ux%u\n", x2_pre.n_rows, x2_pre.n_cols);
            printf("%ux%u\n", x2_post.n_rows, x2_post.n_cols);
            cg_solver.solve(x1, x2_pre, x2_post,
                g_A, delz_A, divider, multiplier, true, 3);
            A_prev = A;
        }*/
        else {
            createMatrixChunks(x1, x2_pre, A, n_half, divider);
            delz_A.zeros(A.n_rows);
            g_A = -g(A);
            
            cg_solver.solve(x1, x2_pre, g_A,
                delz_A, divider, multiplier, true, 3);
            
            A_prev = A;
        }
        
        if (norm(g_A, 2) <= 1) break;

        colvec delz = zeros<vec>(n);
        delz(A) = delz_A;

        double alpha = selectStepSize(A, pos_z, z, delz, delz_A);

        z(A) += delz_A * alpha;
        z.transform([] (double val) { return max(val, 0.); });

        sparsify(z, w, u, l, n_half);
    }

    cout << "Iterations required: " << i << endl;
}
