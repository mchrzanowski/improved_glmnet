#include <limits>
#include <assert.h>
#include "glm.h"
#include "utils.h"

using namespace arma;

GLM::GLM(double eta, uword n_half, uword n) : eta(eta), n_half(n_half), n(n) {
  assert(eta >= 0 && eta <= 1);
  assert(n_half > 0);
  assert(n == 2 * n_half);
}

/* get lambda_max, which is important in
the regularization path calculation. */
double
GLM::maxLambda(){
  // if eta = 0, just pretend it's really small.
  double divisor = eta == 0 ? 1e-4 : eta;
  return norm(g_start, "inf") / divisor;
}

void
GLM::calculateGradient(const colvec &z, double lambda, colvec &g){

  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;

  const colvec g_bias = g_start + lambda * eta;
  const double multiplier = lambda * (1 - eta);
  
  colvec g_half;
  calculateXXw(w, g_half);
  g = g_bias + z * multiplier;
  g.subvec(0, n_half-1) += g_half;
  g.subvec(n_half, n-1) -= g_half;
}

/* solve the elastic net problem for the next lambda in a sequence, where
  the previous lambda value solved for was prev_lambda. use the sequential
  strong rules from [TBFHSTT10] to filter predictors out of the active sets */
size_t
GLM::sequential_solve(colvec &z,
                      double lambda, double prev_lambda,
                      size_t max_iterations){

  colvec u = z.subvec(0, n_half-1).unsafe_col(0);
  colvec l = z.subvec(n_half, n-1).unsafe_col(0);
  colvec w = u - l;

  const colvec yX = g_start.subvec(n_half, n-1);

  colvec val;
  calculateXXw(w, val);

  // whitelisted variables.
  uvec strong = find(abs(yX - val) >= eta * (2 * lambda - prev_lambda));
  strong = join_vert(strong, strong + n_half);

  colvec g;
  uvec active, to_include;

  size_t iters = 0;

  while (true){
    // solve for whitelisted variables.
    iters += solve(z, g, lambda, &strong, max_iterations);

    /* check KKT conditions for strong set.
    is a predictor in the active set that wasn't in the strong set?
    if so, we screwed up. add it to the whitelist
    and re-do the optimization. */
    calculateGradient(z, lambda, g);
    findActiveSet(g, z, active);
    vdifference(active, strong, to_include);

    // solve again, if needed.
    if (to_include.n_rows > 0){
      vunion(strong, to_include, strong);
    }
    else {
      break;
    }
  }
  return iters;
}

/* a solve method that's meant to be called when 
  we need to solve the elastic net problem for only
  one lambda value. this call hides a lot of the internal
  mechanisms we use in the solvers */
size_t
GLM::solve(colvec &z,
            double lambda,
            size_t max_iterations) {
  colvec g;
  return solve(z, g, lambda, NULL, max_iterations);
}

/* evaluate the elastic net function value for a given
  problem instance */
double
GLM::evaluate(const mat &X,
              const colvec &y, const colvec &z,
              double lambda, double eta){

  const colvec w = z.subvec(0, X.n_cols - 1) - 
                    z.subvec(X.n_cols, 2 * X.n_cols - 1);

  return .5 * sum(square(y - X * w))
      + lambda * (eta * norm(w, 1) + .5 * (1 - eta) * sum(square(w)));
}

/* try to find a step size to improve z. return true if 
we were able to find a valid step size, false otherwise */
bool
GLM::update(colvec &z, const uvec &A, const colvec &delz_A,
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

/* project vector to non-negative orthant.
then figure out whether i or n_half + i can be nonzero.
sparsification is really, really useful for speeding up convergence. */
void
GLM::projectAndSparsify(colvec &w, colvec &u, colvec &l){
  static const auto clamp = [](double d) { return d >= 0 ? d : 0; };
  u.transform(clamp);
  l.transform(clamp);
  sparsify(w, u, l);
}

/* Force one of i or n+i to be active. */
void
GLM::sparsify(colvec &w, colvec &u, colvec &l){
  w = u - l; 
  const uvec neg_w = find(w < 0);
  const uvec pos_w = find(w > 0);
  u(neg_w).zeros();
  u(pos_w) = w(pos_w);
  l(pos_w).zeros();
  l(neg_w) = -w(neg_w);
}

/* secret sauce that needs a fat, proper comment */
double
GLM::aggressiveStep(const uvec &A, const vec &eta,
                    colvec &z, const colvec &delz_A, 
                    const colvec &Kz, const colvec &Ku){

  // approximation of gradient at a knot.
  static const auto approx = [](double alpha, double p, double q)
                                { return p * alpha + q; };
  colvec z_A = z(A);

  // our knots are all active indices for which, after a full step size,
  // the values are non-positive. that is, we could in theory
  // clamp these points in this iteration.
  uvec knots = find(delz_A + z_A <= 0);

  // FAILURE.
  if (knots.n_rows == 0) return 0;

  /* as in conservativeStep, the step size that
  clamps index i (which is a knot) is z_i / delz_i. */
  const vec alphas = -z_A(knots) / delz_A(knots);
  const uvec sorted_indices = sort_index(alphas);

  double pi = 0, omega = 0, sigma = 0, c = 0;

  for (uword i = 0; i < sorted_indices.n_rows; i++){
    uword indx = sorted_indices[i];

    double alpha_i = alphas[indx];
    if (alpha_i > 1) break;

    double mu_i = delz_A[knots[indx]];
    double Ku_i = Ku[knots[indx]];
    double Kz_i = Kz[knots[indx]];
    double eta_i = eta[knots[indx]];

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

/* Let "knots" be the indices of the active set for which the gradient is
negative and whose values are positive, is always a safe step size (if
there is still any work to do). That is, these are the indices for which we
can make progress in this iteration. We note that clamping index i (which we
assume is a knot point) means setting alpha to be z_i / delz_i. According to
Bertsekas[82], we can *always* select the smallest step size, hence clamping
the corresponding value for that chosen index. */
double
GLM::conservativeStep(const uvec &A, colvec &z, const colvec &delz_A){

  colvec z_A = z(A);
  const uvec neg_gradient = find(delz_A < 0);
  const uvec pos_z = find(z_A > 0);

  uvec knots;
  vintersection(neg_gradient, pos_z, knots);

  // there is absolutely no work left to do. FAILURE.
  if (knots.n_rows == 0) return 0;

  // alphas are the step lengths required to clamp the values at the knots.
  const vec alphas = z_A(knots) / delz_A(knots);
  double alpha = std::min(-max(alphas), 1.0);
  assert(alpha > 0);

  return alpha;
}

/* find the active set, given the gradient g and the elements of z.
this means look for all elements which have a nonpositive gradient component
or are strictly positive. this basically means that these are the
indices that we can work on trying to decrease in this iteration. */
void
GLM::findActiveSet(const colvec &g, const colvec &z, uvec &A){
  const uvec nonpos_g = find(g <= 0);
  const uvec pos_z = find(z > 0);
  vunion(nonpos_g, pos_z, A);
}

GLM::~GLM(){   
}
