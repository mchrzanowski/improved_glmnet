#include <assert.h>
#include "cg.h"
#include "fat_glm.h"
#include "utils.h"

using namespace arma;
using namespace std;

FatGLM::FatGLM(const mat &X_, const vec &y, const double lambda,
    const double eta) : X(X_), multiplier(lambda * (1 - eta)),
    m(X.n_rows), n(2*X.n_cols), n_half(X.n_cols) {

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);

    const colvec Xy = (y.t() * X).t();
    g_start.zeros(n);
    g_start.subvec(0, n_half-1) = -Xy + lambda * eta;
    g_start.subvec(n_half, n-1) = Xy + lambda * eta;
}

void FatGLM::createMatrixChunks(mat &x1, mat &x2,
    const uvec &A, const uvec &A_prev){

    uword divider = 0;
    while (A(divider) < n_half){
        divider++;
    }
    const uvec A_top = A.subvec(0, divider-1);
    const uvec A_bottom = A.subvec(divider, A.n_rows-1) - n_half;

    x1 = X.cols(A_top);
    x2 = X.cols(A_bottom);
    
}

void FatGLM::solve(colvec &z, const size_t max_iterations){

    CG cg_solver;
    colvec delz_A, g_A;
    const colvec u = z.subvec(0, n_half-1).unsafe_col(0);
    const colvec l = z.subvec(n_half, n-1).unsafe_col(0);
    colvec w = u - l;
    mat x1, x2;
    size_t i;
    uvec A, A_prev;

    colvec u_prev, l_prev;

    int region = 0;

    for (i = 0; i < max_iterations; i++){

        const colvec g_half = ((X * w).t() * X).t();
        colvec g = g_start;
        g.subvec(0, n_half-1) += g_half + u * multiplier;
        g.subvec(n_half, n-1) += -g_half + l * multiplier;
        
        const uvec nonpos_g = find(g <= 0);
        const uvec pos_z = find(z > 0);
        vunion(nonpos_g, pos_z, A);

        if (A.n_rows == 0) break;

        uvec intersect, diff;
        vintersection(A_prev, A, intersect);

        if (intersect.n_rows == A.n_rows) {
            vdifference(A_prev, intersect, diff);
        }

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A.n_rows){
            cg_solver.fatMatrixSolve(x1, x2, g_A,
                delz_A, multiplier, true, 3);
        }
        else if (intersect.n_rows == A.n_rows && diff.n_rows < 30){

            region++;
            uword divider = x1.n_cols;

            uword d = 0, a = 0, b = 0;
            while (d < A_prev.n_rows && A_prev(d) < A(a)){
                if (d < divider){
                    x1.shed_col(0);
                    //cg_solver.getP_top().shed_row(0);
                    //cg_solver.getR_top().shed_row(0);
                }
                else {
                    x2.shed_col(0);
                    //cg_solver.getP_bottom().shed_row(0);
                    //cg_solver.getR_bottom().shed_row(0);
                }
                //g_A.shed_row(0);
                //delz_A.shed_row(0);
                d++;
            }

            while (d < A_prev.n_rows && a < A.n_rows){
                if (A_prev(d) != A(a)){
                    if (d < divider){
                        x1.shed_col(b);
                        //g_A.shed_row(b);
                        //delz_A.shed_row(b);
                        //cg_solver.getP_top().shed_row(b);
                        //cg_solver.getR_top().shed_row(b);
                    }
                    else {
                        x2.shed_col(b);
                        //cg_solver.getP_bottom().shed_row(b);
                        //cg_solver.getR_bottom().shed_row(b);

                        //g_A.shed_row(divider+b);
                        //delz_A.shed_row(divider+b);
                    }
                    
                    if (A_prev(d) < A(a)) {
                        d++;
                    }
                    else {
                        a++;
                        b++;
                    }
                }
                else {
                    d++;
                    a++;
                    b++;
                }

                if (d == divider) b = 0;
            }

            while (d < A_prev.n_rows){
                if (d < divider){
                    x1.shed_col(b);
                    //cg_solver.getP_top().shed_row(b);
                    //cg_solver.getR_top().shed_row(b);
                    //g_A.shed_row(b);
                    //delz_A.shed_row(b);
                }
                else {
                    x2.shed_col(b);
                    //cg_solver.getP_bottom().shed_row(b);
                    //cg_solver.getR_bottom().shed_row(b);
                    //g_A.shed_row(divider+b);
                    //delz_A.shed_row(divider+b);
                }
                d++;
                if (d == divider) b = 0;
            }
            delz_A.zeros(A.n_rows);
            g_A = -g(A);
            cg_solver.fatMatrixSolve(x1, x2, g_A,
                delz_A, multiplier, true, 3);
            A_prev = A;
        }
        else {
            createMatrixChunks(x1, x2, A, A_prev);
            delz_A.zeros(A.n_rows);
            g_A = -g(A);
            cg_solver.fatMatrixSolve(x1, x2, g_A,
                delz_A, multiplier, true, 3);
            A_prev = A;
        }
        
        if (norm(g_A, 2) <= 1) break;
        const uword divider = x1.n_cols;

        const colvec z_A = z(A);
        const colvec z_A_top = z_A.subvec(0, divider-1).unsafe_col(0);
        const colvec z_A_bottom = z_A.subvec(divider, z_A.n_rows-1).unsafe_col(0);

        colvec Kz(A.n_rows);
        fatMultiply(x1, x2, z_A_top, z_A_bottom, multiplier, Kz);

        const colvec delz_A_top = delz_A.subvec(0, divider-1).unsafe_col(0);
        const colvec delz_A_bottom = delz_A.subvec(divider, delz_A.n_rows-1).unsafe_col(0);

        colvec Ku(A.n_rows);
        fatMultiply(x1, x2, delz_A_top, delz_A_bottom, multiplier, Ku);

        bool progress_made = updateBetter(z, A, delz_A, w, u, l, n_half, 
                                            Kz, Ku, g_start(A));
        if (! progress_made) break;

    }

    cout << "Iterations required: " << i << endl;
    cout << "Hits in crit region: " << region << endl;
}