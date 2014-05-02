#pragma once
#include <armadillo>

class GLM {

  public:

     static double crossValidate(const arma::mat &X,
                                  const arma::colvec &y,
                                  arma::colvec &z,
                                  const std::vector<double> &lambdas,
                                  double eta,
                                  double split_ratio,
                                  size_t max_iterations);

      static GLM* makeGLM(const arma::mat &X,
                          const arma::vec &y,
                          double eta,
                          bool unoptimized_solver=false);
     
      virtual ~GLM();

      virtual void solve(arma::colvec &z,
                          double lambda, 
                          size_t max_iterations) = 0;

      static double evaluate(const arma::mat &X,
                              const arma::colvec &y,
                              const arma::colvec &z,
                              double lambda,
                              double eta);

  protected:
      void update(arma::colvec &z,
                  const arma::uvec &A,
                  const arma::colvec &delz_A);

      bool updateBetter(arma::colvec &z,
                        const arma::uvec &A,
                        const arma::colvec &delz_A,
                        const arma::colvec &Kz, 
                        const arma::colvec &Ku, 
                        const arma::vec &eta);

      void projectAndSparsify(arma::colvec &w,
                              arma::colvec &u,
                              arma::colvec &l);

  private:      
      static double clamp(double val);
      
      void sparsify(arma::colvec &w,
                    arma::colvec &u,
                    arma::colvec &l);
      
      double selectStepSize(const arma::uvec &A,
                            arma::colvec &z,
                            const arma::colvec &delz_A);

      double selectImprovedStepSize(const arma::uvec &A,
                                    const arma::vec &eta,
                                    arma::colvec &z,
                                    const arma::colvec &delz_A,
                                    const arma::colvec &Kz,
                                    const arma::colvec &Ku);
};