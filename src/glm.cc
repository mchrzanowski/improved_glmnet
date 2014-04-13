#include "glm.h"
#include "skinny_glm.h"
#include "fat_glm.h"

using namespace arma;

GLM* GLM::makeGLM(const mat &X, const vec &y,
        const double lambda, const double eta){
    const uword m = X.n_rows;
    const uword n = X.n_cols;
    if (n >= 3 * m){   // this seems to work best.
        std::cout << "Created FatGLM class" << std::endl;
        return new FatGLM(X, y, lambda, eta);
    }
    else {
        std::cout << "Created SkinnyGLM class" << std::endl;
        return new SkinnyGLM(X, y, lambda, eta);
    }
}
