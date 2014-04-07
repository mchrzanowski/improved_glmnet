#include <iostream>
#include <armadillo>
#include <set>

using namespace std;
using namespace arma;

template< typename T, template <typename> class ARMA_VECTOR_TYPE >
vector<T> vunion( ARMA_VECTOR_TYPE<T> first, ARMA_VECTOR_TYPE<T> second )
{
    std::vector<T> output ;
    std::set_intersection( first.begin(), first.end(), second.begin(), second.end(),
                           std::back_inserter(output) ) ;
    return output;
}

void init(const mat &A, const vec &b, const vec &x, vec &p, vec &r) {
    r = A * x - b;
    p = -r;
}

int main(int argc, char **argv) {

  size_t size = 10;

  vec x = randn<vec>(size);
  vec p = randn<vec>(size);
  vec r = randn<vec>(size);

  cout << x.t() << endl;
  cout << p.t() << endl;

  const uvec a = find(x < 0);
  const uvec b = find(p >= 0);
  const vector<uword> c = vunion(a, b);

  cout << a.t() << endl;
  cout << b.t() << endl;
  for (uword i : c){
    cout << i << endl;
  }
  
  return 0;
}