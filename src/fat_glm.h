#include "glm.h"

/*
 A class to deal with fat (m < n) matrices
*/
class FatGLM : public GLM {

  public:
    FatGLM(const arma::mat &X, const arma::vec &y, const double eta);
    
    void solve(arma::colvec &z, const double lambda, 
                const size_t max_iterations);

  private:
    void createMatrixChunks(arma::mat &x1, arma::mat &x2, const arma::uvec &A,
                            const arma::uvec &A_prev);

    arma::colvec g_start;
    const arma::mat &X;
    const arma::uword n, n_half;
    const double eta;
};