#pragma once
#include <armadillo>

class GLM {

    public:
        static GLM* makeGLM(const arma::mat &X,const arma::vec &y,
            const double lambda, const double eta,
            bool unoptimizedSolver=false);

        static double clamp(double val);
        
        virtual void solve(arma::colvec &z, const size_t max_iterations) = 0;
        
        virtual ~GLM();

    protected:
        void sparsify(arma::colvec &z, arma::colvec &w, 
            const arma::colvec &u, const arma::colvec &l,
            const arma::uword n_half);
        
        double selectStepSize(const arma::uvec &A,
            arma::colvec &z, const arma::colvec &delz_A);

        double selectImprovedStepSize(const arma::uvec &A,
            const arma::vec &eta, arma::colvec &z,
            const arma::colvec &delz_A, const arma::colvec &Kz,
            const arma::colvec &Ku);

        void update(arma::colvec &z, const arma::uvec A,
            const arma::colvec delz_A, arma::colvec &w,
            const arma::colvec &u, const arma::colvec &l,
            const arma::uword n_half);

        /*void updateBetter(arma::colvec &z, const arma::uvec &A,
            const arma::colvec &delz_A, arma::colvec &w,
            const arma::colvec &u, const arma::colvec &l,
            const arma::uword n_half, const arma::colvec &Kz, 
            const arma::colvec &Ku, const arma::vec &eta);*/
};