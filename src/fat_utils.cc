#include "fat_utils.h"

/* basically the same as below, but y = [y1; y2]; */
void fatMultiply(const mat &A1, const mat &A2, const colvec &x1,
                  const colvec &x2, double multiplier, colvec &y){

  colvec y_top = y.subvec(0, x1.n_rows-1).unsafe_col(0);
  colvec y_bottom = y.subvec(x1.n_rows, y.n_rows-1).unsafe_col(0);

  fatMultiply (A1, A2, x1, x2, multiplier, y_top, y_bottom);
}

/* given that A1 and A2 are fat, perform the multiplication
  y_1 = A1^T * A1 * x1 + A1^T * A2 * x2 + x1 * multiplier
  y_2 = A2^T * A1 * x1 + A2^T * A2 * x2 + x2 * multiplier
  as efficiently as possible. */
void fatMultiply(const mat &A1, const mat &A2, const colvec &x1,
                  const colvec &x2, double multiplier, colvec &y1, colvec &y2){

  const rowvec x1_x_top = (A1 * x1).t();
  const rowvec x2_x_bottom = (A2 * x2).t();

  y1 = (x1_x_top * A1 - x2_x_bottom * A1).t() + x1 * multiplier;
  y2 = (-x1_x_top * A2 + x2_x_bottom * A2).t() + x2 * multiplier;

}