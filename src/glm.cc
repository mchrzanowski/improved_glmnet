#include "glm.h"
#include <assert.h>

using namespace arma;
using namespace std;

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
ARMA_VECTOR_TYPE<T> GLM::vunion( ARMA_VECTOR_TYPE<T> first,
    ARMA_VECTOR_TYPE<T> second) {
    vector<T> output;
    set_union(first.begin(), first.end(), second.begin(), second.end(),
                    back_inserter(output) ) ;
    ARMA_VECTOR_TYPE<T> result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
    return result;
}

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
ARMA_VECTOR_TYPE<T> GLM::vintersection( ARMA_VECTOR_TYPE<T> first,
    ARMA_VECTOR_TYPE<T> second) {
    vector<T> output;
    set_intersection(first.begin(), first.end(), second.begin(), second.end(),
                    back_inserter(output) ) ;
    ARMA_VECTOR_TYPE<T> result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
    return result;
}

GLM::GLM(const mat &X, const vec &y, const double lambda, const double eta)
: eta(eta), lambda(lambda) {

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);
    
    // construct Hessian matrix once and for all.
    const mat XXT = X.t();
    const mat XX = XXT * X;
    const mat XX_I = XX + speye(XX.n_rows, XX.n_cols) * lambda * (1 - eta);
    K = join_vert(join_horiz(XX_I, -XX), join_horiz(-XX, XX_I));

    const vec Xy = XXT * y;
    g_start = join_vert(-Xy, Xy) + lambda * eta;
}

/*
void GLM::create_Kz(vec &g, vec &z){
    uword n = z.n_rows;
    uword n_half = n / 2;
    const vec z_top = z.rows(0, n_half - 1);
    const vec z_bottom = z.rows(n_half, n - 1);
    g.rows(0, n_half-1) = XX_I * z_top - XX * z_bottom;
    g.rows(n_half, n-1) = -XX * z_top + XX_I * z_bottom;
}

void inline GLM::create_K_A(mat &K_A, const uvec &A, const size_t divider){
    const uvec top = find(A < divider);
    const uvec bottom = find(A >= divider);

    const uvec A_top = static_cast<uvec>(A(top));
    const uvec A_bottom = static_cast<uvec>(A(bottom) - divider);

    //const mat system = join_vert(join_horiz(XX_I(top, top), -XX(top, bottom - n_half)), 
    //    join_horiz(-XX(bottom - n_half, top), XX_I(bottom - n_half, bottom - n_half)));

    //K_A = system;
    //cout << K_A.n_rows << "\t" << K_A.n_cols << endl;
    K_A(top, top) = XX_I(A_top, A_top);
    K_A(top, bottom) = -XX(A_top, A_bottom);
    K_A(bottom, top) = -XX(A_bottom, A_top);
    K_A(bottom, bottom) = XX_I(A_bottom, A_bottom);

    //cout << K_A.n_rows << "\t" << K_A.n_cols << endl;
}
*/

void GLM::solve(vec &z, const size_t max_iterations){

    const size_t n_half = z.n_rows / 2;

    CG cg_solver;

    mat K_A;

    size_t i;

    uvec A, A_prev, neg_w(n_half), pos_w(n_half),
        nonpos_g(z.n_rows), pos_z(z.n_rows);

    vec delz(z.n_rows), delz_A, g(g_start.n_rows), g_A, w(n_half);

    for (i = 0; i < max_iterations; i++){

        g = g_start + K * z;

        nonpos_g = find(g <= 0);
        pos_z = find(z > 0);
        A = vunion(nonpos_g, pos_z);
        const size_t A_size = A.n_rows;

        if (A_size == 0) break;

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A_size){
            cg_solver.solve(K_A, g_A, delz_A, false, 3);
        }
        else {
            K_A = K(A, A);  // this is SURPRISINGLY fast.
            delz_A.zeros(A_size);
            g_A = -g(A);
            cg_solver.solve(K_A, g_A, delz_A, true, 3);
        }
        A_prev = A;

        //cout << i << " : " << norm(g_A) << endl;
        if (norm(g_A, 2) <= 1) break;

        delz.zeros();
        delz(A) = delz_A;

        // select step size...
        const uvec neg_delz = A(find(delz_A < 0));
        const uvec D = vintersection(neg_delz, pos_z);
        if (D.n_rows == 0) break;

        const vec alphas = z(D) / delz(D);
        double alpha = min(-max(alphas), 1.0);
        assert(alpha > 0);

        z(A) += delz_A * alpha;
        z.transform([] (double val) { return val > 0 ? val : 0; });

        // force one of indicies of z to be active....
        w = z.subvec(0, n_half-1) - z.subvec(n_half, 2*n_half-1);
        neg_w = find(w < 0);
        pos_w = find(w > 0);
        z(neg_w) *= 0;
        z(pos_w + n_half) *= 0;
        z(neg_w + n_half) = -w(neg_w);
        z(pos_w) = w(pos_w);
    }

    cout << "Iterations required: " << i << endl;
}
