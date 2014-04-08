#include "cg.h"

using namespace arma;

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

/*
	r, p, and vprev can either be reused or recalculated (restart = false/true)
	Note that in this version we use norm(r, 2)^2 as our termination criterion
	so residual_tol should be adjusted accordingly.
*/

void cg_solve2(const mat &A, vec &x, vec &p, vec &r, double *vprev, bool restart, const size_t iterations) {

	double alpha, beta, v;

	if(restart)
	{
		r = A * x - b;
		p = -r;
		vprev[0] = dot(r, r);
	}

    for (size_t i = 0; i < iterations && vprev[0] > RESIDUAL_TOL; i++){
        const vec Ap = A * p;
        
        alpha = vprev[0] / dot(p, Ap);

        x += alpha * p;
        r += alpha * Ap;
		v = dot(r,r);

        beta = v / vprev[0];
        p *= beta;
        p -= r;
		vprev[0] = v;
    }
}