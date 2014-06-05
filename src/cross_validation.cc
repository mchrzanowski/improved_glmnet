#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm_factory.h"

using namespace std;
using namespace arma;

/* Perform cross validation to find a lambda with 
excellent generalization performance on a given dataset.
argc is expected to be 5.
argv[1] = eta
argv[2] = location of A matrix
argv[3] = location of b vector
argv[4] = location of initial z vector
*/
int main(int argc, char **argv){

  assert(argc == 5);
  arma_rng::set_seed(0);

  double eta = strtod(argv[1], NULL);

  mat A; colvec b; colvec z;
  A.load(argv[2], csv_ascii);
  b.load(argv[3], csv_ascii);
  z.load(argv[4], csv_ascii);

  cout << "A Size: " << A.n_rows << " x " << A.n_cols << endl;
  cout << "b Size: " << b.n_rows << endl;
  cout << "z Size: " << z.n_rows << endl;
  cout << "Eta: " << eta << endl;

  wall_clock timer;
  timer.tic();

  double cv_lambda = crossValidate(A, b, z, eta, 0.8);
  
  double time = timer.toc();
  cout << "Runtime: " << time << endl;
  cout << "Chosen lambda: " << cv_lambda << endl;

  return 0;
}
