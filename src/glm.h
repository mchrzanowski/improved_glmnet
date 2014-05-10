#pragma once
#include <armadillo>

using namespace arma;

class GLM {

public:

  virtual ~GLM();

  virtual void solve(colvec &z,
                      double lambda, 
                      size_t max_iterations) = 0;

  virtual double maxLambda() = 0;

  static double evaluate(const mat &X,
                          const colvec &y,
                          const colvec &z,
                          double lambda,
                          double eta);

protected:
  const double G_A_TOL = 0.5;

  bool update(colvec &z,
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
    
  double conservativeStep(const uvec &A,
                          colvec &z,
                          const colvec &delz_A);

  double aggressiveStep(const uvec &A,
                        const vec &eta,
                        colvec &z,
                        const colvec &delz_A,
                        const colvec &Kz,
                        const colvec &Ku);
};
