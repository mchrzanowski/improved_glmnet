#include <iostream>
#include <armadillo>
#include "glm.h"

using namespace std;
using namespace arma;

void test(const mat &A, const vec &b, vec &z, const double lambda, const double eta, const size_t size, const size_t iterations){
    cout << "Iterations: " << iterations << endl;

    GLM g(A, b, lambda, eta);
    cout << "K Created!" << endl;

    g.solve(z, iterations);
    
    vec w = z.subvec(0, size - 1) - z.subvec(size, 2 * size - 1);

    double error = 0.5 * sum(square(b - A * w))
        + lambda * (eta * sum(abs(w)) + 0.5 * (1 - eta) * sum(square(w)));
    cout << "Error: " << error << endl;
}

int main(int argc, char **argv){
    size_t size = 500;
    srand(time(0));

    mat A = randn<mat>(size, size);
    //A += A.t() + size * eye(size, size);
    vec b = randn<vec>(size);
    
    double eta = 0.9;
    double lambda = 0.1;

    vec z = ones<vec>(size * 2) * .1 + randu<vec>(size * 2);

    test(A, b, z, lambda, eta, size, 1000);
   
}
