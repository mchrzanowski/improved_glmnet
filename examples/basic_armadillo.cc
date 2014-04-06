#include <iostream>
#include <armadillo>
 
using namespace std;
using namespace arma;


void init(const mat &A, const vec &b, const vec &x, vec &p, vec &r) {
    r = A * x - b;
    p = -r;
}

int main(int argc, char **argv) {
  
  /*wall_clock timer;
  size_t size = 10000;
  mat A = randn<mat>(size, size);
  mat B = randn<mat>(size, size);

  timer.tic();
  mat C = A * B;
  double n_secs = timer.toc();

  cout << accu(C) << endl;
  cout << "Took: " << n_secs << " seconds." << endl;
  */

  size_t size = 10000;
  mat A = randn<mat>(size, size);
  vec b = randn<vec>(size);
  
  vec x = randn<vec>(size);
  vec p = randn<vec>(size);
  vec r = randn<vec>(size);

  double lol = dot(x, p);
  
  cout << "Value: " << lol << endl;
  cout << "Value: " << dot(x, p) << endl;
  return 0;

  cout << "Before: " << p(0) << "\t" << r(0) << endl;
  cout << &p << "\t" << &r << endl;
  init(A, b, x, p, r);
  cout << "After: " << p(0) << "\t" << r(0) << endl;
  cout << &p << "\t" << &r << endl;

  p(0) = -1;
  cout << p(0) << "\t" << r(0) << endl;


  return 0;
}