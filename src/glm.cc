#include <assert.h>
#include "fat_glm.h"
#include "glm.h"
#include "skinny_glm.h"
#include "utils.h"

using namespace arma;

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

double GLM::selectStepSize(const uvec &A, const uvec &pos_z, const colvec &z,
    const colvec &delz, const colvec &delz_A){
    uvec D;
    const uvec neg_delz = A(find(delz_A < 0));
    vintersection(neg_delz, pos_z, D);
    assert(D.n_rows > 0);

    const vec alphas = z(D) / delz(D);
    double alpha = std::min(-max(alphas), 1.0);
    assert(alpha > 0);

    return alpha;
}

GLM::~GLM(){
    
}
