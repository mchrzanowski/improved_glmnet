#include <iostream>
#include <armadillo>
#include "cg.h"

using namespace std;
using namespace arma;

void test_solver(const mat &A, const mat &b, const vec &x_star, size_t size, size_t iterations){
    vec x = zeros<vec>(size);
    vec p = zeros<vec>(size);
    vec r = zeros<vec>(size);

    cg_init(A, b, x, p, r);
    cg_solve(A, x, p, r, iterations - 1);
    cout << "Iterations: " << iterations << " Error: " << norm(A * x - b, 2) << endl;
}

int main(int argc, char **argv){
    const size_t size = 10000;

    // create symmetric, PD matrix..
    //http://math.stackexchange.com/questions/357980/matlab-code-for-generating-random-symmetric-positive-definite-matrix
    mat A = randn<mat>(size, size);
    A += A.t();
    A += size * eye(size, size);

    const vec b = randn<vec>(size);
    const vec x_star = solve(A, b);

    cout << "Error: " << norm(A * x_star - b, 2) << endl;

    test_solver(A, b, x_star, size, 1);
    test_solver(A, b, x_star, size, 75);
    test_solver(A, b, x_star, size, 80);
}
