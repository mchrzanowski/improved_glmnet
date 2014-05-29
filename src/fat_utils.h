#include <armadillo>

using namespace arma;

/* optimized routines for fat matrices */

void fatMultiply(const mat &A1, const mat &A2,
                  const colvec &x1, const colvec &x2,
                  double multiplier, colvec &y);

void fatMultiply(const mat &A1, const mat &A2,
                  const colvec &x1, const colvec &x2,
                  double multiplier, colvec &y1,
                  colvec &y2);

void fatMultiply(const mat &A1,
                  const colvec &x1,
                  double multiplier,
                  colvec &y1);