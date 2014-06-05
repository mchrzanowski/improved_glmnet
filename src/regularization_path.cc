#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm_factory.h"

using namespace arma;
using namespace std;

/* Regularization path generation.
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
  
  map<double, double> lambda_to_optval;

  wall_clock timer;
  timer.tic();

  regularizationPath(A, b, z, lambda_to_optval, eta);

  double time = timer.toc();
  cout << "Runtime: " << time << endl;

  for (auto &kv : lambda_to_optval){
    cout << "Lambda: " << kv.first << "\tOptVal: " << kv.second << endl;
  }

  return 0;
}
