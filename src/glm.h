#include <armadillo>
#include "cg.h"

class GLM {

public:
    GLM(const arma::mat &X, const arma::vec &y,const double lambda, const double eta);
    void solve(arma::vec &z, const double epsilon, const size_t max_iterations);
    
    template<typename T, template <typename> class ARMA_VECTOR_TYPE>
    ARMA_VECTOR_TYPE<T> vunion(ARMA_VECTOR_TYPE<T> first, ARMA_VECTOR_TYPE<T> second);

    template<typename T, template <typename> class ARMA_VECTOR_TYPE>
    ARMA_VECTOR_TYPE<T> vintersection( ARMA_VECTOR_TYPE<T> first, ARMA_VECTOR_TYPE<T> second);

private:
    arma::mat X;
    arma::mat K;
    arma::vec y;
    double eta, lambda;
};