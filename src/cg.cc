#include "cg.h"
#include "utils.h"

using namespace arma;

arma::vec& CG::getP_top(){
    return p_top;
}

arma::vec& CG::getP_bottom(){
    return p_bottom;
}

arma::vec& CG::getR_top(){
    return r_top;
}

arma::vec& CG::getR_bottom(){
    return r_bottom;
}

void CG::fatMatrixSolve(const mat &x1, 
    const mat &x2,
    const vec &b,
    vec &x,
    const double multiplier,
    const bool restart,
    const size_t iterations){

    const uword half = x1.n_cols;

    vec x_top = x.subvec(0, half-1).unsafe_col(0);
    vec x_bottom = x.subvec(half, x.n_rows-1).unsafe_col(0);

    const vec b_top = b.subvec(0, half-1).unsafe_col(0);
    const vec b_bottom = b.subvec(half, x.n_rows-1).unsafe_col(0);

    if (restart) {
        // vector transpose is dirt cheap, but
        // avoid transposing matrices at ALL COST: that's *really* expensive.
        // ((x1 * x_top)^T * x1)^T == (x1^T * x1) * x_top
        // ((x2 * -x_bottom)^T * x1)^T == -x1^T * x2 * x_bottom

        const rowvec x1_x_top = (x1 * x_top).t();
        const rowvec x2_x_bottom = (x2 * x_bottom).t();

        r_top = (x1_x_top * x1 - x2_x_bottom * x1).t()
            - b_top + x_top * multiplier;

        r_bottom = (-x1_x_top * x2 + x2_x_bottom * x2).t()
            - b_bottom + x_bottom * multiplier;
        
        p_top = -r_top;
        p_bottom = -r_bottom;
        prev_r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
    }

    for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){

        colvec Ap_top, Ap_bottom;
        fatMultiply(x1, x2, p_top, p_bottom, multiplier,  Ap_top, Ap_bottom);

        const double alpha = prev_r_sq_sum / 
            (dot(p_top, Ap_top) + dot(p_bottom, Ap_bottom));

        x_top += alpha * p_top;
        x_bottom += alpha * p_bottom;
        
        r_top += alpha * Ap_top;
        r_bottom += alpha * Ap_bottom;
        
        const double r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
        const double beta = r_sq_sum / prev_r_sq_sum;

        p_top *= beta;
        p_top -= r_top;
        
        p_bottom *= beta;
        p_bottom -= r_bottom;
        
        prev_r_sq_sum = r_sq_sum;
    }

}

void CG::solve(const mat &A, const vec &b, vec &x,
    const bool restart, const size_t iterations) {

    if (restart) {
        r = A * x - b;
        p = -r;
        prev_r_sq_sum = dot(r, r);
    }
    
    for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
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

void CG::skinnyMatrixSolve(const mat &x1, 
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

    for (size_t i = 0; i < iterations && prev_r_sq_sum > RESIDUAL_TOL; i++){
        const colvec Ap_top = x1 * p_top + x2 * p_bottom;
        const colvec Ap_bottom = (p_top.t() * x2).t() + x4 * p_bottom;

        const double alpha = prev_r_sq_sum / 
            (dot(p_top, Ap_top) + dot(p_bottom, Ap_bottom));

        x_top += alpha * p_top;
        x_bottom += alpha * p_bottom;
        
        r_top += alpha * Ap_top;
        r_bottom += alpha * Ap_bottom;

        const double r_sq_sum = dot(r_top, r_top) + dot(r_bottom, r_bottom);
        const double beta = r_sq_sum / prev_r_sq_sum;
        
        p_top *= beta;
        p_top -= r_top;

        p_bottom *= beta;
        p_bottom -= r_bottom;

        prev_r_sq_sum = r_sq_sum;
    }
}
