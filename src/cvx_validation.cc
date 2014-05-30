#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm_factory.h"

using namespace std;
using namespace arma;

static void run(GLM *g, const mat &A, const colvec &b, colvec &z,
                double lambda, double eta){
  
  wall_clock timer;
  timer.tic();
  g->solve(z, lambda);
  double time = timer.toc();
  double error = GLM::evaluate(A, b, z, lambda, eta);

  cout << "Lambda: " << lambda << endl;
  cout << "Error: " << error << endl;
  cout << "Runtime: " << time << " seconds. " << endl;
}

/*
CVX  verifier.
Will solve the elastic net optimization problem for a given A, b, z, eta, and
lambda value. You should then solve the problem in CVX and make sure the
optimal values are close to each other.
argc is expected to be 7.
argv[1] = lambda
argv[2] = eta
argv[3] = location of A matrix
argv[4] = location of b vector
argv[5] = location of initial z vector
argv[6] = use the unoptimized solver?
*/
int main(int argc, char **argv){

  assert(argc == 7);
  arma_rng::set_seed(0);

  double lambda = strtod(argv[1], NULL);
  double eta = strtod(argv[2], NULL);
  bool use_stupid_solver = strtol(argv[6], NULL, 10) > 0;

  mat A; colvec b; colvec z;
  A.load(argv[3], csv_ascii);
  b.load(argv[4], csv_ascii);
  z.load(argv[5], csv_ascii);

  lambda = 0.5 * norm(A.t() * b, "inf") / eta;

  cout << A.n_rows << "x" << A.n_cols 
    << "\t" << b.n_rows << "\t" 
    << z.n_rows << endl;

  cout << "Lambda: " << lambda << endl;
  cout << "Eta: " << eta << endl;


  GLM *g = makeGLM(A, b, eta, use_stupid_solver);
  run(g, A, b, z, lambda, eta);

  delete g; 
}
