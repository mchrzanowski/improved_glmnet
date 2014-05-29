#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(){
    const size_t size = 30000;
    mat A = randn<mat>(size, size);
    cout << accu(A) << endl;
    mat B = A.submat(0, 0, 14999, size - 1);
    cout << B.n_rows << 'x' << B.n_cols << endl;
    cout << accu(B) << endl;
}