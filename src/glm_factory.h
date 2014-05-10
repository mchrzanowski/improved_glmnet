#pragma once
#include <armadillo>
#include "glm.h"

using namespace arma;

GLM* makeGLM(const mat &X,
              const vec &y,
              double eta,
              bool unoptimized_solver=false);

double crossValidate(const mat &X,
                      const colvec &y,
                      colvec &z,
                      double eta,
                      double split_ratio,
                      size_t max_iterations);
