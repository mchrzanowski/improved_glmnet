#include <limits>
#include <assert.h>
#include "glm.h"
#include "utils.h"

using namespace arma;

GLM::GLM(double eta) : eta(eta) {}

/* project to non-negative orphant */
double GLM::clamp(double val){
  return std::max(val, 0.);
}

double GLM::maxLambda(){
  assert(eta > 0);
  return 0.9 * norm(g_start, "inf") / eta;
}

/* evlaute the elastic net function value for a given
  problem instance */
double GLM::evaluate(const mat &X, const colvec &y, const colvec &z,
                      double lambda, double eta){

  const colvec w = z.subvec(0, X.n_cols - 1) - 
                    z.subvec(X.n_cols, 2 * X.n_cols - 1);

  return .5 * sum(square(y - X * w))
      + lambda * (eta * norm(w, 1) + .5 * (1 - eta) * sum(square(w)));
}

/* try to find a step size to improve z. return true if 
we were able to find a valid step size, false otherwise */
bool GLM::update(colvec &z, const uvec &A, const colvec &delz_A,
  const colvec &Kz, const colvec &Ku, const vec &eta){

  // try to get the largest step possible.
  double alpha = aggressiveStep(A, eta, z, delz_A, Kz, Ku);
  
  if (alpha == 0) {
    // failed to get a step length. fall back to the Bertsekas step size,
    // which is guaranteed to make progress...
    alpha = conservativeStep(A, z, delz_A);
    //..if there was still work to do. guess not. STOP EVERYTHING.
    if (alpha == 0) return false;
  }
  
  z(A) += delz_A * alpha;
  return true;
}

/* project vector to non-negative orphant.
then figure out whether i or n_half + i can be nonzero. */
void GLM::projectAndSparsify(colvec &w, colvec &u, colvec &l){
  u.transform(clamp);
  l.transform(clamp);
  sparsify(w, u, l);
}

/* Force one of i or n+i to be active. */
void GLM::sparsify(colvec &w, colvec &u, colvec &l){
  w = u - l; 
  const uvec neg_w = find(w < 0);
  const uvec pos_w = find(w > 0);
  u(neg_w).zeros();
  u(pos_w) = w(pos_w);
  l(pos_w).zeros();
  l(neg_w) = -w(neg_w);
}

/* secret sauce that needs a fat, proper comment */
double GLM::aggressiveStep(const uvec &A, const vec &eta,
                            colvec &z, const colvec &delz_A, 
                            const colvec &Kz, const colvec &Ku){

  // approximation of gradient at a knot.
  const auto approx = [](double alpha, double p, double q)
                        { return p * alpha + q; };
  colvec z_A = z(A);

  // find all indices for which, after a full step size,
  // we're nonpositive.
  uvec D = find(delz_A + z_A <= 0);

  // FAILURE.
  if (D.n_rows == 0) return 0;

  const vec alphas = -z_A(D) / delz_A(D);
  const uvec sorted_indices = sort_index(alphas);

  double pi = 0, omega = 0, sigma = 0, c = 0;

  for (uword i = 0; i < sorted_indices.n_rows; i++){
    uword indx = sorted_indices[i];

    double alpha_i = alphas[indx];
    if (alpha_i > 1) break;

    double mu_i = delz_A[D[indx]];
    double Ku_i = Ku[D[indx]];
    double Kz_i = Kz[D[indx]];
    double eta_i = eta[D[indx]];

    pi -= 2 * mu_i * Ku_i;
    omega += -mu_i * (Kz_i + eta_i) + alpha_i * mu_i * Ku_i;
    sigma += mu_i;
    c += alpha_i * mu_i;

    double p = pi + sigma * sigma;
    double q = omega - sigma * c;

    if (i < sorted_indices.n_rows - 1
        && approx(alphas[sorted_indices[i + 1]], p, q) < 0){
      continue;
    }

    if (approx(alpha_i, p, q) >= 0){
      return alpha_i; // guaranteed to be <= 1.
    }
    else {
      return std::min(1., -p / q);
    }
  }

  return 1;
}

/* Bertsekas[82] shows the first knot is always a safe step size (if
there is still any work to do) */
double GLM::conservativeStep(const uvec &A, colvec &z, const colvec &delz_A){

  colvec z_A = z(A);
  const uvec neg_gradient = find(delz_A < 0);
  const uvec pos_z = find(z_A > 0);

  uvec D;
  vintersection(neg_gradient, pos_z, D);

  // there is absolutely no work left to do. FAILURE.
  if (D.n_rows == 0) return 0;

  const vec alphas = z_A(D) / delz_A(D);
  double alpha = std::min(-max(alphas), 1.0);
  assert(alpha > 0);

  return alpha;
}

/* find the active set, given the gradient g and the elements of z.
this means look for all elements which have a nonpositive gradient
or are strictly positive. this basically means that these are the
indices that we can work on in this iteration. */
void GLM::findActiveSet(const colvec &g, const colvec &z, uvec &A){
  const uvec nonpos_g = find(g <= 0);
  const uvec pos_z = find(z > 0);
  vunion(nonpos_g, pos_z, A);
}

GLM::~GLM(){   
}
