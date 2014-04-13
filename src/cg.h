#include <armadillo>

class CG {

public:
    void solve(const arma::mat &x1, 
        const arma::mat &x2,
        const arma::mat &x4,
        const arma::vec &b,
        arma::vec &x,
        const arma::uword half, 
        const bool restart,
        const size_t iterations=3);

    void solve(const arma::mat &x1_pre, 
        const arma::mat &x1_post, 
        const arma::mat &x2_pre,
        const arma::mat &x2_post,
        const arma::mat &x4_pre,
        const arma::mat &x4_post,
        const arma::vec &b,
        arma::vec &x,
        const arma::uword half, 
        const double multiplier,
        const bool restart,
        const size_t iterations=3);

    /*void solve(const arma::mat &A, const arma::vec &b, arma::vec &x,
        const bool restart, const size_t iterations=3);*/

private:
    const double RESIDUAL_TOL = 1e-3;
    double prev_r_sq_sum;
    arma::vec p_top, p_bottom, r_top, r_bottom; //p, r;
};
