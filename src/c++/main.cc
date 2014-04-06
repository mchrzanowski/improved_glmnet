#include <iostream>
#include <armadillo>
#include "cg.h"

using namespace std;
using namespace arma;

void test_solver(const mat &A, const mat &b, const vec &x_star,
size_t size, size_t iterations){
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
}

int main(int argc, char **argv){
    const size_t size = 20000;

    // create symmetric, PD matrix..
    mat A = randn<mat>(size, size);
    A += A.t();
    A += size * eye(size, size);

    const vec b = randn<vec>(size);

    wall_clock timer;
    timer.tic();
    const vec x_star = solve(A, b);
    double duration = timer.toc();
    cout << "Error: " << norm(A * x_star - b, 2)
        << "\tRuntime: " << duration << endl;

    test_solver(A, b, x_star, size, 80);

}
