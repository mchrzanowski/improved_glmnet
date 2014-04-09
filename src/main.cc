#include <iostream>
#include <armadillo>
#include "glm.h"

using namespace std;
using namespace arma;

void test(const mat &A, const vec &b, vec &z, const double lambda,
const double eta, const size_t size, const size_t iterations){
    cout << "Iterations: " << iterations << endl;

    GLM g(A, b, lambda, eta);
    cout << "K Created!" << endl;

    g.solve(z, iterations);
    
    vec w = z.subvec(0, size - 1) - z.subvec(size, 2 * size - 1);

    double error = 0.5 * sum(square(b - A * w))
        + lambda * (eta * sum(abs(w)) + 0.5 * (1 - eta) * sum(square(w)));
    cout << "Error: " << error << endl;
}

void cg_test(){
    size_t size = 8000;
    mat A = randn<mat>(size, size);
    A += A.t() + size * eye(size, size);

    vec b = randn<vec>(size);

    CG cg_solver;
    vec x = randn<vec>(size);
    cg_solver.solve(A, b, x, true, 100);
    cout << "CG: " << norm(A * x - b, 2) << endl;
}

void glm_test(int size=500){

    mat A = randn<mat>(size, size);
    //A += A.t() + size * eye(size, size);
    vec b = randn<vec>(size);
    
    double eta = 0.9;
    double lambda = 30;

    vec z = ones<vec>(size * 2) * .1 + randu<vec>(size * 2);

    test(A, b, z, lambda, eta, size, 1000);
}

int main(int argc, char **argv){
    srand(time(0));
    if (argc > 1){
        int size = atoi(argv[1]);
        glm_test(size);
    }
    else {
        glm_test();
    }
}
