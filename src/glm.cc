#include "glm.h"
#include <assert.h>

using namespace arma;
using namespace std;

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void GLM::vunion(ARMA_VECTOR_TYPE<T> &first,
    ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
    vector<T> output;
    set_union(first.begin(), first.end(), second.begin(), second.end(),
                    back_inserter(output) ) ;
    result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void GLM::vintersection(ARMA_VECTOR_TYPE<T> &first,
    ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
    vector<T> output;
    set_intersection(first.begin(), first.end(), second.begin(), second.end(),
                    back_inserter(output) ) ;
    result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void GLM::vdifference(ARMA_VECTOR_TYPE<T> &first,
    ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
    vector<T> output;
    set_difference(first.begin(), first.end(), second.begin(), second.end(),
                    back_inserter(output) ) ;
    result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

GLM::GLM(const mat &X, const vec &y, const double lambda, const double eta)
: eta(eta), lambda(lambda) {

    assert(eta > 0 && eta <= 1);
    assert(lambda > 0);
    
    const mat XXT = X.t();
    XX = XXT * X;

    const colvec Xy = XXT * y;
    g_start.zeros(2*y.n_rows);
    g_start.subvec(0, y.n_rows-1) = -Xy + lambda * eta;
    g_start.subvec(y.n_rows, 2*y.n_rows-1) = Xy + lambda * eta;
}

void GLM::create_K_A(mat &K_A, const uvec &A, const size_t divider){
    


}


void GLM::solve(colvec &z, const size_t max_iterations){

    const size_t n_half = z.n_rows / 2;

    CG cg_solver;

    mat K_A;

    size_t i;

    uvec A, A_prev, D, neg_w(n_half), pos_w(n_half),
        neg_delz, nonpos_g(z.n_rows), pos_z(z.n_rows);

    colvec delz(z.n_rows), delz_A, g(g_start.n_rows), g_A;

    const colvec u = z.subvec(0, n_half-1).unsafe_col(0);
    const colvec l = z.subvec(n_half, 2*n_half-1).unsafe_col(0);
    colvec w = u - l;

    mat x1, x2, x4;
    uword bottom = 0;

    for (i = 0; i < max_iterations; i++){

        const colvec g_half = XX * w;

        g = g_start;
        g.subvec(0, n_half-1) += g_half + u * lambda * (1 - eta);
        g.subvec(n_half, 2*n_half-1) += -g_half + l * lambda * (1 - eta);

        nonpos_g = find(g <= 0);
        pos_z = find(z > 0);
        vunion(nonpos_g, pos_z, A);
        const size_t A_size = A.n_rows;

        if (A_size == 0) break;

        //uvec lol;
        //vintersection(A_prev, A, lol);

        if (A.n_rows == A_prev.n_rows && accu(A == A_prev) == A_size){
            //cg_solver.solve(K_A, g_A, delz_A, false, 3);
            cg_solver.solve(x1, x2, x4, g_A, delz_A, bottom, true, 3);
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
            
            //wall_clock timer;
            //mat old;

            //timer.tic();
            const uvec top = find(A < n_half);
            bottom = top(top.n_rows-1) + 1;

            const uvec A_top = static_cast<uvec>(A(top));
            const uvec A_bottom = static_cast<uvec>(A.subvec(bottom, A.n_rows-1) - n_half);

            x1 = XX(A_top, A_top) + speye(A_top.n_rows, A_top.n_rows) * lambda * (1-eta);
            x2 = -XX(A_top, A_bottom);
            x4 = XX(A_bottom, A_bottom) + speye(A_bottom.n_rows, A_bottom.n_rows) * lambda * (1-eta);
            //double new_way = timer.toc();

            //timer.tic();
            //K_A = K(A, A);
            //double old_way = timer.toc();

            //cout << "Matrix creation: " << new_way << "\t" << old_way << "\t" << new_way - old_way << endl;


            delz_A.zeros(A_size);
            g_A = -g(A);
            //timer.tic();
            //cg_solver.solve(K_A, g_A, delz_A, true, 3);
            //old_way = timer.toc();

            //vec new_delz_A = zeros<vec>(A_size);
            //timer.tic();
            cg_solver.solve(x1, x2, x4, g_A, delz_A, bottom, true, 3);
            //new_way = timer.toc();

            //cout << "CG: " << new_way << "\t" << old_way << "\t" << new_way - old_way << endl;
            //cout << "CG NOrm: " << norm(new_delz_A - delz_A) << endl;


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

        // force one of indicies of z to be active....
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
