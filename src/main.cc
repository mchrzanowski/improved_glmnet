#include <iostream>
#include <armadillo>
#include "glm.h"

using namespace std;
using namespace arma;

/*void test_solver(const mat &A, const mat &b, const vec &x_star, size_t size, size_t iterations){
    vec x = zeros<vec>(size);
    vec p = zeros<vec>(size);
    vec r = zeros<vec>(size);

    wall_clock timer;
    timer.tic();
    cg_init(A, b, x, p, r);
    cg_solve(A, x, p, r, iterations - 1);
    double duration = timer.toc();
    cout << "Iterations: " << iterations
        << " Error: " << norm(A * x - b, 2) 
        << "\tRuntime: " << duration << endl;
}*/

void test(const mat &A, const vec &b, const double lambda, const double eta, const size_t size, const size_t iterations){
    cout << "Iterations: " << iterations << endl;
    vec z = randn<vec>(size * 2);
    
    vec w = z.subvec(0, size - 1) - z.subvec(size, 2 * size - 1);
    double pre_error = 0.5 * sum(square(b - A * w))
        + lambda * (eta * sum(abs(w)) + 0.5 * (1 - eta) * sum(square(w)));
    cout << "Post-Error: " << pre_error << endl;

    GLM g(A, b, lambda, eta);
    cout << "K Created!" << endl;
    g.solve(z, 1e-8, iterations);
    
    w = z.subvec(0, size - 1) - z.subvec(size, 2 * size - 1);
    double post_error = 0.5 * sum(square(b - A * w))
        + lambda * (eta * sum(abs(w)) + 0.5 * (1 - eta) * sum(square(w)));
    cout << "Post-Error: " << post_error << endl;
}

int main(int argc, char **argv){
    const size_t size = 1000;

    // create symmetric, PD matrix..
    mat A = randn<mat>(size, size);
    A += A.t();
    A += size * eye(size, size);

    const vec b = randn<vec>(size);
    
    double eta = 0.5;
    double lambda = 0.1;

    test(A, b, lambda, eta, size, 100);
    test(A, b, lambda, eta, size, 500);
    test(A, b, lambda, eta, size, 900);
}
