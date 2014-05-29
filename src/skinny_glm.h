#include "glm.h"

/*
A class to deal with skinny (m > n) matrices
*/
class SkinnyGLM : public GLM {

public:
  SkinnyGLM(const mat &X, const vec &y, double eta);

  void solve(colvec &z, double lambda, size_t max_iterations=0);

private:
  void createMatrixChunks(mat &x1, mat &x2, mat &x4,
                            const uvec &A, const uvec &A_prev,
                            double multiplier);

  mat XX;
  const uword n, n_half;
};