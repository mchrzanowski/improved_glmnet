#include <assert.h>
#include "fat_glm.h"
#include "glm.h"
#include "skinny_glm.h"
#include "utils.h"

using namespace arma;

double GLM::clamp(double val){
    return std::max(val, 0.);
}

GLM* GLM::makeGLM(const mat &X, const vec &y,
        const double lambda, const double eta){
    if (X.n_cols >= 3 * X.n_rows){   // this seems to work best.
        std::cout << "Created FatGLM class" << std::endl;
        return new FatGLM(X, y, lambda, eta);
    }
    else {
        std::cout << "Created SkinnyGLM class" << std::endl;
        return new SkinnyGLM(X, y, lambda, eta);
    }
}

void GLM::update(colvec &z, const uvec A, const colvec delz_A,
    colvec &w, const colvec &u, const colvec &l, const uword n_half){
    const double alpha = selectStepSize(A, z, delz_A);
    z(A) += delz_A * alpha;
    z.transform(clamp);
    sparsify(z, w, u, l, n_half);
}

void GLM::sparsify(colvec &z, colvec &w, const colvec &u,
    const colvec &l, const uword n_half){
    // force one of indices of z to be active....
    w = u - l; 
    const uvec neg_w = find(w < 0);
    const uvec pos_w = find(w > 0);
    z(neg_w).zeros();
    z(pos_w + n_half).zeros();
    z(neg_w + n_half) = -w(neg_w);
    z(pos_w) = w(pos_w);
}

double GLM::selectStepSize(const uvec &A,
    colvec &z, const colvec &delz_A){

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
