#include "cg.h"
#include <armadillo>
#include <iostream>

using namespace arma;

int main(){
    size_t size = 5000;
    mat A = randn<mat>(size, size);
    vec b = randn<vec>(size);

    vec x_star = solve(A, b);

    CG cg_solver;
    vec x = randn<vec>(size);
    cg_solver.solve(A, b, x, true, 200);

    cout << "Perfect: " << norm(A * x_star - b, 2) << endl;
    cout << "CG: " << norm(A * x - b, 2) << endl;

    return 0;
}