#include "cg.h"

using namespace arma;

void CG::solve(const mat &x1, 
            const mat &x2_pre,
            const mat &x2_post,
            const vec &b,
            vec &x,
            const uword half, 
            const double multiplier,
            const bool restart,
            const size_t iterations){

    vec x_top = x.subvec(0, half-1).unsafe_col(0);
    vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0);

    const vec b_top = b.subvec(0, half-1).unsafe_col(0);
    const vec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);

    if (restart) {
        r_top = ((x1 * x_top).t() * x1).t() + 
            x2_post * (x2_pre * -x_bottom) - b_top + x_top * multiplier;
        
        r_bottom = ((-x_top.t() * x2_post) * x2_pre).t() + 
            + ((x2_pre * x_bottom).t() * x2_pre).t()
            - b_bottom + x_bottom * multiplier;
        
        p_top = -r_top;
        p_bottom = -r_bottom;
        prev_r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
    }

    for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
        const colvec Ap_top = ((x1 * p_top).t() * x1).t() + 
            x2_post * (x2_pre * -p_bottom) + p_top * multiplier;

        const colvec Ap_bottom = ((-p_top.t() * x2_post) * x2_pre).t() 
            + ((x2_pre * p_bottom).t() * x2_pre).t() + p_bottom * multiplier;

        double alpha = prev_r_sq_sum / 
            (dot(p_top, Ap_top) + dot(p_bottom, Ap_bottom));

        x_top += alpha * p_top;
        x_bottom += alpha * p_bottom;
        r_top += alpha * Ap_top;
        r_bottom += alpha * Ap_bottom;
        double r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);

        double beta = r_sq_sum / prev_r_sq_sum;
        p_top *= beta;
        p_bottom *= beta;
        p_top -= r_top;
        p_bottom -= r_bottom;
        prev_r_sq_sum = r_sq_sum;
    }

}

/*void CG::solve(const mat &A, const vec &b, vec &x,
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
}*/

void CG::solve(const mat &x1, 
        const mat &x2,
        const mat &x4,
        const vec &b,
        vec &x,
        const uword half, 
        const bool restart,
        const size_t iterations){

    vec x_top = x.subvec(0, half-1).unsafe_col(0);
    vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0);

    const vec b_top = b.subvec(0, half-1).unsafe_col(0);
    const vec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);

	if (restart) {
		r_top = x1 * x_top + x2 * x_bottom - b_top;
        r_bottom = (x_top.t() * x2).t() + x4 * x_bottom - b_bottom;
		p_top = -r_top;
        p_bottom = -r_bottom;
		prev_r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
	}

    double alpha, beta, r_sq_sum;
    colvec Ap_top(half), Ap_bottom(half);

    for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
        Ap_top = x1 * p_top + x2 * p_bottom;
        Ap_bottom = (p_top.t() * x2).t() + x4 * p_bottom;

        alpha = prev_r_sq_sum / 
            (dot(p_top, Ap_top) + dot(p_bottom, Ap_bottom));

        x_top += alpha * p_top;
        x_bottom += alpha * p_bottom;
        r_top += alpha * Ap_top;
        r_bottom += alpha * Ap_bottom;
        r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);

        beta = r_sq_sum / prev_r_sq_sum;
        p_top *= beta;
        p_bottom *= beta;
        p_top -= r_top;
        p_bottom -= r_bottom;
        prev_r_sq_sum = r_sq_sum;
    }
}
