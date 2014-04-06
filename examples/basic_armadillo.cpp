#include <iostream>
#include <armadillo>
 
using namespace std;
using namespace arma;
 
int main(int argc, char **argv) {
  
  wall_clock timer;
  size_t size = 10000;
  mat A = randn(size, size);
  mat B = randn(size, size);

  timer.tic();
  mat C = A * B;
  double n_secs = timer.toc();

  cout << accu(C) << endl;
  cout << "Took: " << n_secs << " seconds." << endl;

  return 0;
}