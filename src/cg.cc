#include "cg.h"

void init(const mat& A, const vec& b, const vec& x, vec& p, vec& r) {
    r = A * x - b;
    p = -r;
}

void solve(const mat& A, vec& x, vec& p, vec& r, size_t iterations=3) {
    for (size_t i = 0; i < iterations; i++){
        vec Ap = A * p;
        
        double old_r_sum = dot(r, r);
        double alpha = old_r_sum / dot(p, Ap);
        x += alpha * p;
        r += alpha * Ap;
        
        // p_(k+1) = -r_(k+1) + beta_k * p_k
        double beta = dot(r, r) / old_r_sum;
        p *= beta;
        p -= r;
    }
}