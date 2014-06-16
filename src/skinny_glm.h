#include "glm.h"

/*
A class to deal with skinny (m > n) matrices
*/
class SkinnyGLM : public GLM {

public:
  SkinnyGLM(const mat &X, const vec &y, double eta);

  void calculateXXw(const colvec &w, colvec &ret);

  size_t solve(colvec &z,
                double lambda,
                size_t max_iterations=0);

  size_t solve(colvec &z, colvec &g,
                double lambda,
                const uvec *whitelisted=NULL,
                size_t max_iterations=0);

private:
  void createMatrixChunks(mat &x1, mat &x2, mat &x4,
                            const uvec &A, const uvec &A_prev,
                            double multiplier);

  mat XX;
};