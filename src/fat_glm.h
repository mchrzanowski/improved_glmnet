#include "glm.h"

/*
 A class to deal with fat (m < n) matrices
*/
class FatGLM : public GLM {

  public:
    FatGLM(const mat &X, const vec &y, const double eta);
    
    void solve(colvec &z, double lambda, size_t max_iterations);

  private:
    void createMatrixChunks(mat &x1, mat &x2, const uvec &A,
                            const uvec &A_prev);

    colvec g_start;
    const mat &X;
    const uword n, n_half;
    const double eta;
};