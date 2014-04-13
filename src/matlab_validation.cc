#include <iostream>
#include <armadillo>
#include "glm.h"
#include <assert.h>

using namespace std;
using namespace arma;

int main(int argc, char **argv){

    wall_clock timer;

    mat A; colvec b; colvec z;

    A.load("_A", csv_ascii);
    b.load("_b", csv_ascii);
    z.load("_z", csv_ascii);

    cout << A.n_rows << "x" << A.n_cols 
        << "\t" << b.n_rows << "\t" 
        << z.n_rows << endl;

    assert(argc == 4);
    double lambda = strtod(argv[1], NULL);
    double eta = strtod(argv[2], NULL);
    long iterations = strtol(argv[3], NULL, 10);

    cout << lambda << "\t" << eta << "\t" << iterations << endl;

    timer.tic();
    GLM *g = GLM::makeGLM(A, b, lambda, eta);
    g->solve(z, iterations);
    
    double time = timer.toc();
    vec w = z.subvec(0, A.n_cols - 1) - z.subvec(A.n_cols, 2 * A.n_cols - 1);

    double error = 0.5 * sum(square(b - A * w))
        + lambda * (eta * norm(w, 1) + 0.5 * (1 - eta) * sum(square(w)));

    cout << "Error: " << error << endl;
    cout << "Runtime: " << time << " seconds. " << endl;
}