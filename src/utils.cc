#include "utils.h"

using namespace arma;

void fatMultiply(const mat &A1, const mat &A2, const colvec &x1,
const colvec &x2, const double multiplier, colvec &y){

  const rowvec x1_x_top = (A1 * x1).t();
  const rowvec x2_x_bottom = (A2 * x2).t();

  y.subvec(0, x1.n_rows-1) = (x1_x_top * A1 - x2_x_bottom * A1).t()
      + x1 * multiplier;

  y.subvec(x1.n_rows, y.n_rows-1) = (-x1_x_top * A2 + x2_x_bottom * A2).t()
      + x2 * multiplier;
}

void fatMultiply(const mat &A1, const mat &A2, const colvec &x1,
const colvec &x2, const double multiplier, colvec &y1, colvec &y2){

  const rowvec x1_x_top = (A1 * x1).t();
  const rowvec x2_x_bottom = (A2 * x2).t();

  y1 = (x1_x_top * A1 - x2_x_bottom * A1).t() + x1 * multiplier;
  y2 = (-x1_x_top * A2 + x2_x_bottom * A2).t() + x2 * multiplier;

}