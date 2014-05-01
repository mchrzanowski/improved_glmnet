#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm.h"

using namespace std;
using namespace arma;

static void run(GLM *g, const mat &A, const colvec &b, colvec &z,
                double lambda, double eta, size_t iters){
  wall_clock timer;
  timer.tic();
  g->solve(z, lambda, iters);
  double time = timer.toc();
  vec w = z.subvec(0, A.n_cols - 1) - z.subvec(A.n_cols, 2 * A.n_cols - 1);

  double error = 0.5 * sum(square(b - A * w))
      + lambda * (eta * norm(w, 1) + 0.5 * (1 - eta) * sum(square(w)));

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

  std::vector<double> lambdas = std::vector<double>();
  lambdas.push_back(0.03125);
  lambdas.push_back(0.0625);
  lambdas.push_back(0.125);
  lambdas.push_back(0.25);
  lambdas.push_back(0.5);
  lambdas.push_back(1);
  lambdas.push_back(2);
  lambdas.push_back(4);
  lambdas.push_back(8);
  lambdas.push_back(16);
  double best = GLM::crossValidate(A, b, z, lambdas, eta, 0.8, 12000);
  cout << "Best lambda: " << best << endl;

  cout << "Eta: " << eta << endl
    << "Max Iters: " << iterations << endl;

  GLM *g = GLM::makeGLM(A, b, eta, use_stupid_solver);
  run(g, A, b, z, best / 2, eta, iterations);
  run(g, A, b, z, best, eta, iterations);
  run(g, A, b, z, lambda / 2, eta, iterations);
  run(g, A, b, z, lambda, eta, iterations);
  run(g, A, b, z, lambda * 2, eta, iterations);

  delete g;
  
}