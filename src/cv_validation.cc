#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm_factory.h"

using namespace std;
using namespace arma;

/*
argc is expected to be 7.
argv[1] = eta
argv[2] = max iterations
argv[3] = location of A matrix
argv[4] = location of b vector
argv[5] = location of initial z vector
*/
int main(int argc, char **argv){

  assert(argc == 6);
  arma_rng::set_seed(0);

  double eta = strtod(argv[1], NULL);
  long iterations = strtol(argv[2], NULL, 10);

  mat A; colvec b; colvec z;
  A.load(argv[3], csv_ascii);
  b.load(argv[4], csv_ascii);
  z.load(argv[5], csv_ascii);

  cout << A.n_rows << "x" << A.n_cols 
    << "\t" << b.n_rows << "\t" 
    << z.n_rows << endl;

  cout << "Eta: " << eta << endl
    << "Max Iters: " << iterations << endl;

  regularizationPath(A, b, z, eta, iterations);

}