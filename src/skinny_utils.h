#include <armadillo>

using namespace arma;

void skinnyMultiply(const mat &x1, const mat &x2, const mat &x4,
                    const colvec &x_top, const colvec &x_bottom,
                    colvec &y);

void skinnyMultiply(const mat &x1, const mat &x2, const mat &x4,
                    const colvec &x_top, const colvec &x_bottom,
                    colvec &y1, colvec &y2);

void skinnyMultiply(const mat &A,
                    const colvec &x,
                    colvec &y);