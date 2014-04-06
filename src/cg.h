#include <armadillo>

void init(const mat& A, const vec& b, const vec& x, vec& p, vec& r);
void solve(const mat& A, vec& x, vec& p, vec& r, size_t iterations);