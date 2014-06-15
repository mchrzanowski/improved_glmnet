#include "glm.h"

using namespace arma;

/*
 A class to deal with fat (m < n) matrices
*/
class FatGLM : public GLM {

public:
  FatGLM(const mat &X, const vec &y, const double eta);
  
  size_t solve(colvec &z,
                double lambda,
                size_t max_iterations=0);

  size_t sequential_solve(colvec &z,
                          double lambda, double prev_lambda,
                          size_t max_iterations=0);

  size_t solve(colvec &z, colvec &g,
                double lambda,
                const uvec *blacklisted=NULL,
                size_t max_iterations=0);

private:
  void createMatrixChunks(mat &x1, mat &x2,
                          const uvec &A, const uvec &A_prev);

  void calculateGradient(const colvec &z, double lambda, colvec &g);

  const mat &X;
  const uword n_half, n;
};