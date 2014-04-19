#include <armadillo>
#include <assert.h>
#include <iostream>
#include "glm.h"

using namespace std;
using namespace arma;

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

    cout << lambda << "\t" << eta << "\t" << iterations << endl;

    wall_clock timer;
    timer.tic();
    GLM *g = GLM::makeGLM(A, b, lambda, eta, use_stupid_solver);
    g->solve(z, iterations);
    
    delete g;
    double time = timer.toc();
    vec w = z.subvec(0, A.n_cols - 1) - z.subvec(A.n_cols, 2 * A.n_cols - 1);

    double error = 0.5 * sum(square(b - A * w))
        + lambda * (eta * norm(w, 1) + 0.5 * (1 - eta) * sum(square(w)));

    cout << "Error: " << error << endl;
    cout << "Runtime: " << time << " seconds. " << endl;
}