#pragma once
#include <armadillo>

using namespace arma;

class GLM {

public:

  static double crossValidate(const mat &X,
                              const colvec &y,
                              colvec &z,
                              const std::vector<double> &lambdas,
                              double eta,
                              double split_ratio,
                              size_t max_iterations);

  static GLM* makeGLM(const mat &X,
                      const vec &y,
                      double eta,
                      bool unoptimized_solver=false);
   
  virtual ~GLM();

  virtual void solve(colvec &z,
                      double lambda, 
                      size_t max_iterations) = 0;

  static double evaluate(const mat &X,
                          const colvec &y,
                          const colvec &z,
                          double lambda,
                          double eta);

protected:
  const double G_A_TOL = 0.5;

  void update(colvec &z,
              const uvec &A,
              const colvec &delz_A);

  void updateBetter(colvec &z,
                    const uvec &A,
                    const colvec &delz_A,
                    const colvec &Kz, 
                    const colvec &Ku, 
                    const vec &eta);

  void projectAndSparsify(colvec &w,
                          colvec &u,
                          colvec &l);

  void findActiveSet(const colvec &g, const colvec &z, uvec &A);

private:
  static double clamp(double val);
    
  void sparsify(colvec &w,
                colvec &u,
                colvec &l);
    
  double selectStepSize(const uvec &A,
                        colvec &z,
                        const colvec &delz_A);

  double selectImprovedStepSize(const uvec &A,
                                const vec &eta,
                                colvec &z,
                                const colvec &delz_A,
                                const colvec &Kz,
                                const colvec &Ku);
};
