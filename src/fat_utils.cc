#include "fat_utils.h"

/* basically the same as below, but y = [y1; y2]; */
void fatMultiply(const mat &A1, const mat &A2,
                  const colvec &x1, const colvec &x2,
                  double multiplier,
                  colvec &y){

  if (A1.n_rows > 0 && A1.n_cols > 0 && A2.n_rows > 0 && A2.n_cols > 0){
    colvec y_top = y.subvec(0, x1.n_rows-1).unsafe_col(0);
    colvec y_bottom = y.subvec(x1.n_rows, y.n_rows-1).unsafe_col(0);
    fatMultiply (A1, A2, x1, x2, multiplier, y_top, y_bottom);
  }
  else if (A1.n_rows == 0 || A1.n_cols == 0){
    colvec y_top;
    colvec y_bottom = y.subvec(x1.n_rows, y.n_rows-1).unsafe_col(0);
    fatMultiply (A1, A2, x1, x2, multiplier, y_top, y_bottom);
  }
  else {
    colvec y_top = y.subvec(0, x1.n_rows-1).unsafe_col(0);
    colvec y_bottom;
    fatMultiply (A1, A2, x1, x2, multiplier, y_top, y_bottom);
  }
  
}

/* given that A1 and A2 are fat, perform the multiplication
  y_1 = A1^T * A1 * x1 + A1^T * A2 * x2 + x1 * multiplier
  y_2 = A2^T * A1 * x1 + A2^T * A2 * x2 + x2 * multiplier
  as efficiently as possible. */
void fatMultiply(const mat &A1, const mat &A2,
                  const colvec &x1, const colvec &x2,
                  double multiplier,
                  colvec &y1, colvec &y2){

  // vector transpose is dirt cheap, but
  // avoid transposing matrices at ALL COST: that's *really* expensive.
  // ((x1 * x_top)^T * x1)^T == (x1^T * x1) * x_top
  // ((x2 * -x_bottom)^T * x1)^T == -x1^T * x2 * x_bottom
  if (A1.n_rows > 0 && A1.n_cols > 0 && A2.n_rows > 0 && A2.n_cols > 0){
    const rowvec x1_x_top = (A1 * x1).t();
    const rowvec x2_x_bottom = (A2 * x2).t();
    y1 = (x1_x_top * A1 - x2_x_bottom * A1).t() + x1 * multiplier;
    y2 = (-x1_x_top * A2 + x2_x_bottom * A2).t() + x2 * multiplier;
  }
  else if (A1.n_rows == 0 || A1.n_cols == 0){
    const rowvec x2_x_bottom = (A2 * x2).t();
    y2 = (x2_x_bottom * A2).t() + x2 * multiplier;
  }
  else {
    const rowvec x1_x_top = (A1 * x1).t();
    y1 = (x1_x_top * A1).t() + x1 * multiplier;
  }

}


