#pragma once
#include <armadillo>

class GLM {

  public:
    static GLM* makeGLM(const arma::mat &X,const arma::vec &y,
        const double lambda, const double eta);
    virtual void solve(arma::colvec &z, const size_t max_iterations) = 0;
    virtual ~GLM();
};