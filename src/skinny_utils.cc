#include "skinny_utils.h"

// wrapper.
void skinnyMultiply(const mat &x1, const mat &x2, const mat &x4,
                    const colvec &x_top, const colvec &x_bottom,
                    colvec &y) {
  colvec y_top = y.subvec(0, x1.n_rows-1).unsafe_col(0);
  colvec y_bottom = y.subvec(x1.n_rows, y.n_rows-1).unsafe_col(0);
  skinnyMultiply(x1, x2, x4, x_top, x_bottom, y_top, y_bottom);
}

void skinnyMultiply(const mat &x1, const mat &x2, const mat &x4,
                    const colvec &x_top, const colvec &x_bottom,
                    colvec &y1, colvec &y2) {
  // avoid tranposing matrices.
  y1 = x1 * x_top + x2 * x_bottom;
  y2 = (x_top.t() * x2).t() + x4 * x_bottom;
}