#include "cg.h"
#include "utils.h"

using namespace arma;

void CG::solve(const mat &A, const vec &b, vec &x,
    const bool restart, const size_t iterations) {

    if (restart) {
        r = A * x - b;
        p = -r;
        prev_r_sq_sum = dot(r, r);
    }
    
    for (size_t i = 0; i < iterations; i++){
        const vec Ap = A * p;
        const double alpha = prev_r_sq_sum / dot(p, Ap);

        x += alpha * p;
        r += alpha * Ap;
        const double r_sq_sum = dot(r, r);

        const double beta = r_sq_sum / prev_r_sq_sum;
        p *= beta;
        p -= r;
        prev_r_sq_sum = r_sq_sum;
    }

}
