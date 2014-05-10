#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm_factory.h"

using namespace std;
using namespace arma;

static void run(GLM *g, const mat &A, const colvec &b, colvec &z,
                double lambda, double eta, size_t iters){
  
  wall_clock timer;
  timer.tic();
  g->solve(z, lambda, iters);
  double time = timer.toc();
  double error = GLM::evaluate(A, b, z, lambda, eta);

  cout << "Lambda: " << lambda << endl;
  cout << "Error: " << error << endl;
  cout << "Runtime: " << time << " seconds. " << endl;
}

/*
MATLAB verifier
argc is expected to be 8.
argv[1] = lambda
argv[2] = eta
argv[3] = max iterations
argv[4] = location of A matrix
argv[5] = location of b vector
argv[6] = location of initial z vector
argv[7] = use the unoptimized solver?
*/
int main(int argc, char **argv){

  assert(argc == 8);
  arma_rng::set_seed(0);

  double lambda = strtod(argv[1], NULL);
  double eta = strtod(argv[2], NULL);
  long iterations = strtol(argv[3], NULL, 10);
  bool use_stupid_solver = strtol(argv[7], NULL, 10) > 0;

  mat A; colvec b; colvec z;
  A.load(argv[4], csv_ascii);
  b.load(argv[5], csv_ascii);
  z.load(argv[6], csv_ascii);

  cout << A.n_rows << "x" << A.n_cols 
    << "\t" << b.n_rows << "\t" 
    << z.n_rows << endl;

  cout << "Eta: " << eta << endl
    << "Max Iters: " << iterations << endl;

  GLM *g = makeGLM(A, b, eta, use_stupid_solver);
  run(g, A, b, z, lambda, eta, iterations);

  delete g; 
}
