#include "cg.h"

using namespace arma;

/*
void cg_init(const mat &A, const vec &b, const vec &x, vec &p, vec &r) {
    r = A * x - b;
    p = -r;
}

void cg_solve(const mat &A, vec &x, vec &p, vec &r, const size_t iterations) {
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
*/

void CG::solve(const mat &A, const vec &b, vec &x,
    const bool restart, const size_t iterations) {

	double alpha, beta, r_sq_sum;

	if (restart) {
		r = A * x - b;
		p = -r;
		prev_r_sq_sum = dot(r, r);
	}

    for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
        const vec Ap = A * p;
        alpha = prev_r_sq_sum / dot(p, Ap);

        x += alpha * p;
        r += alpha * Ap;
		r_sq_sum = dot(r, r);

        beta = r_sq_sum / prev_r_sq_sum;
        p *= beta;
        p -= r;
		prev_r_sq_sum = r_sq_sum;
    }
}
