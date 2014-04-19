#include "glm.h"

/*
 A class to deal with skinny (m > n) matrices
*/
class SkinnyGLM : public GLM {

public:
    SkinnyGLM(const arma::mat &X, const arma::vec &y,const double lambda,
        const double eta);
    void solve(arma::colvec &z, const size_t max_iterations);

private:
    arma::uword createMatrixChunks(arma::mat &x1, arma::mat &x2,
        arma::mat &x4, const arma::uvec &A);

    arma::colvec g_start;
    arma::mat XX;
    const double multiplier;
    const arma::uword m, n, n_half;
};