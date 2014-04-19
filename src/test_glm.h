#include "glm.h"

/* A class that isn't optimized for speed but
    rather for readability and pedagogy.
    Used to test things out before implementing
    them in the optimized solvers */
class TestGLM : public GLM {

public:
    TestGLM(const arma::mat &X, const arma::vec &y,
        const double lambda, const double eta);
    void solve(arma::colvec &z, const size_t max_iterations);

private:
    arma::colvec g_start;
    arma::mat XX, K;
    double eta, lambda;
};