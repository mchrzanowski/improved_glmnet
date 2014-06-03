#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm_factory.h"

using namespace std;
using namespace arma;

/*
CVX  verifier.
Will solve the elastic net optimization problem for a given A, b, z, and eta.
You should then solve the problem in CVX and make sure the
optimal values are close to each other.
As you probably don't know a good lambda value, we'll choose one for you.
Just plug in the value we report into CVX.

argc is expected to be 6.
argv[1] = eta
argv[2] = location of A matrix
argv[3] = location of b vector
argv[4] = location of initial z vector
argv[5] = use the unoptimized solver?
*/
int main(int argc, char **argv){

  assert(argc == 6);
  arma_rng::set_seed(0);

  double eta = strtod(argv[1], NULL);
  bool use_stupid_solver = strtol(argv[5], NULL, 10) > 0;

  mat A; colvec b; colvec z;
  A.load(argv[2], csv_ascii);
  b.load(argv[3], csv_ascii);
  z.load(argv[4], csv_ascii);

  cout << "A Size: " << A.n_rows << " x " << A.n_cols << endl;
  cout << "b Size: " << b.n_rows << endl;
  cout << "z Size: " << z.n_rows << endl;
  cout << "Eta: " << eta << endl;

  GLM *g = makeGLM(A, b, eta, use_stupid_solver);
  
  // pick a reasonable lambda.
  double lambda = 0.25 * g->maxLambda();
  
  cout << "Lambda: " << lambda << endl;

  wall_clock timer;
  timer.tic();
  
  g->solve(z, lambda);
  
  double time = timer.toc();
  double error = GLM::evaluate(A, b, z, lambda, eta);

  cout << "Error: " << error << endl;
  cout << "Runtime: " << time << " seconds. " << endl;

  delete g; 
  return 0;
}
