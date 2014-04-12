#include <armadillo>

class CG {

public:
    void solve(const arma::mat &A, const arma::vec &b, arma::vec &x,
        const bool restart, const size_t iterations=3);

private:
    const double RESIDUAL_TOL = 1e-3;
    double prev_r_sq_sum;
    arma::vec p, r;
};
