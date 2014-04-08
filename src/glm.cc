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
    const mat XX = X.t() * X;
    const mat XX_I = XX + eye(XX.n_rows, XX.n_cols) * lambda * (1 - eta);
    K = join_vert(join_horiz(XX_I, -XX), join_horiz(-XX, XX_I));

    const vec Xy = X.t() * y;
    g_start = join_vert(-Xy, Xy) + ones<vec>(Xy.n_rows * 2) * lambda * eta;

}

void GLM::solve(vec &z, const size_t max_iterations){
    
    vec g_A;
    vec p;
    vec r;
    vec delz_A;
    uvec A_prev;
    mat K_A;

    vec delz = zeros<vec>(z.n_rows);
    CG cg_solver;
    
    for (size_t i = 0; i < max_iterations; i++){
        const vec g = g_start + K * z;

        // ask about this...
        if (norm(g, 2) < 1e-18) break;

        const uvec nonpos_g = find(g <= 0);
        const uvec pos_z = find(z > 0);
        const uvec A = GLM::vunion(nonpos_g, pos_z);
        const size_t A_size = A.n_rows;

        if (A_size == 0) break;

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A_size){
            cg_solver.solve(K_A, g_A, delz_A, false, 3);
        }
        else {
            K_A = K(A, A);
            delz_A = zeros<vec>(A_size);
            g_A = -g(A);
            cg_solver.solve(K_A, g_A, delz_A, true, 3);
        }
        A_prev = A;

        delz.fill(0);
        delz(A) += delz_A;

        // select step size...
        const uvec neg_delz = find(delz < 0);
        const uvec D = GLM::vintersection(neg_delz, pos_z);
        if (D.n_rows == 0) break;

        const vec alphas = -z(D) / delz(D);
        double alpha = min(alphas);

        assert(alpha > 0);
        delz(A) *= alpha;

        z += delz;
        z.transform([] (double val) { return val > 0 ? val : 0; });
    }
}