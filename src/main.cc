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

int main(int argc, char **argv){
    const size_t size = 100;

    // create symmetric, PD matrix..
    mat A = randn<mat>(size, size);
    A += A.t();
    A += size * eye(size, size);

    const vec b = randn<vec>(size);

    vec z = randn<vec>(size * 2);

    GLM g(A, b, 0.1, 0.5);
    cout << "K Created!" << endl;
    g.solve(z, 1e-8, 10000);

    vec x = z.subvec(0, size - 1) - z.subvec(size, 2 * size - 1);

    cout << norm(A * x - b, 2) << endl;

}
