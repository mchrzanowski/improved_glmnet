#include "glm_factory.h"
#include "fat_glm.h"
#include "skinny_glm.h"
#include "test_glm.h"

/* create a GLM solver instance.
  pick one based on whether the input data matrix 
  is skinny, fat, or whether we should use the basic solver */
GLM*
makeGLM(const mat &X,
        const vec &y,
        double eta,
        bool unoptimized_solver){
  if (unoptimized_solver){
    std::cout << "Using TestGLM solver." << std::endl;
    return new TestGLM(X, y, eta);
  }
  else if (X.n_cols >= 3 * X.n_rows){   // works well in practice.
    std::cout << "Using FatGLM solver." << std::endl;
    return new FatGLM(X, y, eta);
  }
  else {
    std::cout << "Using SkinnyGLM solver." << std::endl;
    return new SkinnyGLM(X, y, eta);
  }
}

/* perform cross validation, where we 
  take in a list of lambda values, a ratio for
  training/validation splitting, and problem data,
  and return the lambda that has the best value
  on the validation set */
double
crossValidate(const mat &X,
              const colvec &y,
              colvec &z,
              double eta,
              double split_ratio,
              size_t max_iterations){
  // permute data.
  uvec permute = shuffle(linspace<uvec>(0, X.n_rows-1, X.n_rows));
  uword last_tr_sample = split_ratio * X.n_rows;

  const mat X_train = X.rows(permute.subvec(0, last_tr_sample-1));
  const colvec y_train = y(permute.subvec(0, last_tr_sample-1));

  const mat X_test = X.rows(permute.subvec(last_tr_sample, permute.n_rows-1));
  const colvec y_test = y(permute.subvec(last_tr_sample, permute.n_rows-1));

  GLM *g = makeGLM(X_train, y_train, eta);
  double best_lambda = -1;
  double best_val = std::numeric_limits<double>::max();

  // get initial lambda
  double max_lambda = g->maxLambda();
  double lambda = max_lambda;
  for (unsigned i = 0; i < 100; i++){
    g->solve(z, lambda, NULL, max_iterations);
    double val = GLM::evaluate(X_test, y_test, z, lambda, eta);
    if (val < best_val){
      best_lambda = lambda;
      best_val = val;
    }
    lambda *= 0.9545486;
  }
  delete g;
  return best_lambda;
}

/* calculate regularization path */
void
regularizationPath(const mat &X,
                    const colvec &y,
                    colvec &z,
                    std::map<double, double> &lambda_to_optval,
                    std::map<double, colvec> &lambda_to_zstar,
                    double eta,
                    size_t max_iterations){
  
  GLM *g = makeGLM(X, y, eta);
  double max_lambda = g->maxLambda();
  double lambda = max_lambda;
  double prev_lambda = -1;
  for (unsigned i = 0; i < 100; i++){
    if (i == 0){
      g->solve(z, lambda, NULL, max_iterations);
    }
    else {
      g->sequential_solve(z, lambda, prev_lambda, max_iterations);
    }
    double value = GLM::evaluate(X, y, z, lambda, eta);
    lambda_to_optval[lambda] = value;
    lambda_to_zstar[lambda] = z;
    prev_lambda = lambda;
    lambda *= 0.9545486;
  }
  delete g;
}
