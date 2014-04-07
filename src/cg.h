#include <armadillo>

const double RESIDUAL_TOL = 1e-13;

void cg_init(const arma::mat &A, const arma::vec &b, const arma::vec &x, arma::vec &p, arma::vec &r);
void cg_solve(const arma::mat &A, arma::vec &x, arma::vec &p, arma::vec &r, const size_t iterations=2);