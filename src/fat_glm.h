#include <armadillo>
#include "glm.h"

/*
 A class to deal with fat (m < n) matrices
*/
class FatGLM : public GLM {

public:
    FatGLM(const arma::mat &X, const arma::vec &y, const double lambda,
        const double eta);
    void solve(arma::colvec &z, const size_t max_iterations);

private:
    arma::uword createMatrixChunks(arma::mat &x1,
        arma::mat &x2, const arma::uvec &A);

    arma::colvec g_start;
    const arma::mat &X;
    const double multiplier;
    const arma::uword m, n, n_half;
};