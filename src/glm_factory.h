#pragma once
#include <armadillo>
#include <map>
#include "glm.h"

using namespace arma;

GLM*
makeGLM(const mat &X,
        const vec &y,
        double eta,
        bool unoptimized_solver=false);

double
crossValidate(const mat &X,
              const colvec &y,
              colvec &z,
              double eta,
              double split_ratio,
              size_t max_iterations=0);

void
regularizationPath(const mat &X,
                    const colvec &y,
                    colvec &z,
                    std::map<double, double> &lambda_to_optval,
                    std::map<double, colvec> &lambda_to_zstar,
                    double eta,
                    size_t max_iterations=0);
