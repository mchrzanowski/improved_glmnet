#include <armadillo>
#include "cg.h"

class GLM {

public:
    GLM(const arma::mat &X, const arma::vec &y,const double lambda, const double eta);
    void solve(arma::colvec &z, const size_t max_iterations);

private:
    template<typename T, template <typename> class ARMA_VECTOR_TYPE>
    void vunion(ARMA_VECTOR_TYPE<T> &first, ARMA_VECTOR_TYPE<T> &second,
        ARMA_VECTOR_TYPE<T> &result);

    template<typename T, template <typename> class ARMA_VECTOR_TYPE>
    void vintersection(ARMA_VECTOR_TYPE<T> &first, ARMA_VECTOR_TYPE<T> &second,
        ARMA_VECTOR_TYPE<T> &result);

    template<typename T, template <typename> class ARMA_VECTOR_TYPE>
    void vdifference(ARMA_VECTOR_TYPE<T> &first, ARMA_VECTOR_TYPE<T> &second,
        ARMA_VECTOR_TYPE<T> &result);

    //void create_Kz(arma::vec &g, arma::vec &z);
    //void create_K_A(arma::mat &K_A, const arma::uvec &A, const size_t n_half);

    arma::colvec g_start;
    arma::mat XX, K;
    double eta, lambda;
};