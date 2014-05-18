#include "skinny_utils.h"

/* various multiplication routines needed to multiply
K * x
where:
  K = [A1; A2; A2^T; A4],
  x = [x_top; x_bottom],
  and K is skinny.

Note that we run into degenerate cases when A1 or A4 are empty
(in which case A2 is always empty. For details, see
SkinnyGLM::createMatrixChunks).

*/

// wrapper.
void skinnyMultiply(const mat &A1, const mat &A2, const mat &A4,
                    const colvec &x_top, const colvec &x_bottom,
                    colvec &y) {
  // if both matrices are non-empty, do the normal routine.
  if (A1.n_rows > 0 && A1.n_cols > 0 && A4.n_rows > 0 && A4.n_cols > 0){
    colvec y_top = y.subvec(0, A1.n_rows-1).unsafe_col(0);
    colvec y_bottom = y.subvec(A1.n_rows, y.n_rows-1).unsafe_col(0);
    skinnyMultiply(A1, A2, A4, x_top, x_bottom, y_top, y_bottom);
  }
  // if A1 is empty, then only calculate y_bottom assuming A2 is empty.
  else if (A1.n_rows == 0 || A1.n_cols == 0) {
    colvec y_bottom = y.subvec(A1.n_rows, y.n_rows-1).unsafe_col(0);
    skinnyMultiply(A4, x_bottom, y_bottom);
  }
  // the opposite case when A4 is empty (and hence A2 is also empty)
  else {
    colvec y_top = y.subvec(0, A1.n_rows-1).unsafe_col(0);
    skinnyMultiply(A1, x_top, y_top);
  }
}

/* given that A1, A2, and A4 are skinny, perform the multiplication
  y_1 = A1 * x_top + A2 * x_bottom
  y_2 = A2 ^T * x_top + A4 * x_bottom
  as efficiently as possible. */
void skinnyMultiply(const mat &A1, const mat &A2, const mat &A4,
                    const colvec &x_top, const colvec &x_bottom,
                    colvec &y1, colvec &y2) {
  // avoid transposing matrices.
  if (A1.n_rows > 0 && A1.n_cols > 0 && A4.n_rows > 0 && A4.n_cols > 0){
    y1 = A1 * x_top + A2 * x_bottom;
    y2 = (x_top.t() * A2).t() + A4 * x_bottom;
  }
  else if (A1.n_rows == 0 || A1.n_cols == 0){
    skinnyMultiply(A4, x_bottom, y2);
  }
  else {
    skinnyMultiply(A1, x_top, y1);
  }
}

/* special case when one of A1 or A4 is empty. Hence, A2 is empty */
void skinnyMultiply(const mat &A,
                    const colvec &x,
                    colvec &y) {
  y = A * x;
}
