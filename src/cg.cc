#include "cg.h"

using namespace arma;

void cg_init(const mat &A, const vec &b, const vec &x, vec &p, vec &r) {
    r = A * x - b;
    p = -r;
}

void cg_solve(const mat &A, vec &x, vec &p, vec &r, const size_t iterations=3) {
    for (size_t i = 0; i < iterations && norm(r, 2) > RESIDUAL_TOL; i++){
        const vec Ap = A * p;
        
        double old_r_sum = dot(r, r);
        double alpha = old_r_sum / dot(p, Ap);

        x += alpha * p;
        r += alpha * Ap;

        double beta = dot(r, r) / old_r_sum;
        p *= beta;
        p -= r;
    }
}