#include <limits>
#include <assert.h>
#include "fat_glm.h"
#include "glm.h"
#include "skinny_glm.h"
#include "test_glm.h"
#include "utils.h"

using namespace arma;

/* perform cross validation, where we 
  take in a list of lambda values, a ratio for
  training/validation splitting, and problem data,
  and return the lambda that has the best error
  on the validation set */
double GLM::crossValidate(const mat &X,
                          const colvec &y,
                          colvec &z,
                          const std::vector<double> &lambdas,
                          double eta,
                          double split_ratio,
                          size_t max_iterations){
  // permute data.
  uvec permute = shuffle(linspace<uvec>(0, X.n_rows - 1, X.n_rows));
  uword last_tr_sample = split_ratio * X.n_rows;

  const mat X_train = X.rows(permute.subvec(0, last_tr_sample - 1));
  const colvec y_train = y.rows(permute.subvec(0, last_tr_sample - 1));

  const mat X_test = X.rows(permute.subvec(last_tr_sample, 
                                            permute.n_rows - 1));
  const colvec y_test = y.rows(permute.subvec(last_tr_sample,
                                              permute.n_rows - 1));

  GLM *g = makeGLM(X_train, y_train, eta);
  double best_lambda = -1;
  double best_error = std::numeric_limits<double>::max();
  wall_clock total;
  total.tic();
  for (double lambda: lambdas){
    wall_clock timer;
    timer.tic();
    g->solve(z, lambda, max_iterations);
    double time = timer.toc();
    double error = evaluate(X_test, y_test, z, lambda, eta);
    std::cout << "Lambda: " << lambda << "\t" << "Error: " << error << 
      "\t" << "Runtime: " << time << std::endl;
    if (error < best_error){
      best_lambda = lambda;
      best_error = error;
    }
  }
  double total_time = total.toc();
  std::cout << "Total Runtime: " << total_time << std::endl;
  return best_lambda;
}

double GLM::approximation(double alpha, double p, double q){
  return p * alpha + q;
}

/* projcet to non-negative orphant */
double GLM::clamp(double val){
  return std::max(val, 0.);
}

/* evlaute the elastic net error value for a given
  problem instance */
double GLM::evaluate(const mat &X, const colvec &y, const colvec &z,
                      double lambda, double eta){

  const colvec w = z.subvec(0, X.n_cols - 1) - 
                    z.subvec(X.n_cols, 2 * X.n_cols - 1);

  return 0.5 * sum(square(y - X * w))
      + lambda * (eta * norm(w, 1) + 0.5 * (1 - eta) * sum(square(w)));
}

GLM* GLM::makeGLM(const mat &X, const vec &y, double eta,
                  bool unoptimizedSolver){

  if (unoptimizedSolver){
    std::cout << "Created TestGLM class" << std::endl;
    return new TestGLM(X, y, eta);
  }
  else if (X.n_cols >= 3 * X.n_rows){   // works well in practice.
    std::cout << "Created FatGLM class" << std::endl;
    return new FatGLM(X, y, eta);
  }
  else {
    std::cout << "Created SkinnyGLM class" << std::endl;
    return new SkinnyGLM(X, y, eta);
  }
}

void GLM::update(colvec &z, const uvec &A, const colvec &delz_A){
  const double alpha = selectStepSize(A, z, delz_A);
  z(A) += delz_A * alpha;
}

bool GLM::updateBetter(colvec &z, const uvec &A, const colvec &delz_A,
  const colvec &Kz, const colvec &Ku, const vec &eta){
  const double alpha = selectImprovedStepSize(A, eta, z, delz_A, Kz, Ku);
  if (alpha == 0) return false;
  z(A) += delz_A * alpha;
  return true;
}

/* project vector elmements to non-negative orphant.
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

double GLM::selectImprovedStepSize(const uvec &A, const vec &eta,
  colvec &z, const colvec &delz_A, const colvec &Kz, const colvec &Ku){

  colvec z_A = z(A);
  
  // find all indices for which, after a full step size,
  // we're nonpositive.
  uvec D = find(delz_A + z_A <= 0);

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
        && approximation(alphas[sorted_indices[i + 1]], p, q) < 0){
      continue;
    }

    if (GLM::approximation(alpha_i, p, q) >= 0){
      return alpha_i; // guaranteed to be <= 1.
    }
    else {
      return std::min(1., -p / q);
    }
  }

  return 1;
}

double GLM::selectStepSize(const uvec &A, colvec &z, const colvec &delz_A){

  colvec z_A = z(A);
  const uvec neg_gradient = find(delz_A < 0);
  const uvec pos_z = find(z_A > 0);

  uvec D;
  vintersection(neg_gradient, pos_z, D);
  assert(D.n_rows > 0);

  const vec alphas = z_A(D) / delz_A(D);
  double alpha = std::min(-max(alphas), 1.0);
  assert(alpha > 0);

  return alpha;
}

GLM::~GLM(){   
}
