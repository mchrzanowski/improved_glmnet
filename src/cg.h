#include <armadillo>

class CG {

public:
    void skinnyMatrixSolve(const arma::mat &x1, 
        const arma::mat &x2,
        const arma::mat &x4,
        const arma::vec &b,
        arma::vec &x,
        const arma::uword half, 
        const bool restart,
        const size_t iterations=3);

    void fatMatrixSolve(const arma::mat &x1, 
        const arma::mat &x2_pre,
        const arma::vec &b,
        arma::vec &x,
        const double multiplier,
        const bool restart,
        const size_t iterations=3);

    void solve(const arma::mat &A,
        const arma::vec &b, arma::vec &x,
        const bool restart, const size_t iterations=3);

    arma::vec& getP_top();
    arma::vec& getP_bottom();

    arma::vec& getR_top();
    arma::vec& getR_bottom();

private:
    const double RESIDUAL_TOL = 1e-3;
    double prev_r_sq_sum;
    arma::vec p_top, p_bottom, r_top, r_bottom, r, p;
};
