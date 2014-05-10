#include <armadillo>

using namespace arma;

/* take union of two vectors. close to:
http://www.cplusplus.com/forum/general/70687 */
template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void vunion(const ARMA_VECTOR_TYPE<T> &first,
  const ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
  std::vector<T> output;
  std::set_union(first.begin(), first.end(),
                  second.begin(), second.end(),
                  back_inserter(output));
  result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

/* vector element-wise intersection. close to:
http://www.cplusplus.com/forum/general/70687 */
template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void vintersection(const ARMA_VECTOR_TYPE<T> &first,
  const ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
  std::vector<T> output;
  std::set_intersection(first.begin(), first.end(),
                      second.begin(), second.end(),
                      back_inserter(output));
  result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

/* vector element-wise difference */
template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void vdifference(const ARMA_VECTOR_TYPE<T> &first,
  const ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
  std::vector<T> output;
  std::set_difference(first.begin(), first.end(),
                      second.begin(), second.end(),
                      back_inserter(output));
  result = conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

/* search for the element index for which
  v(index - 1) < target && v(index) >= target
  obviously assumes that the vector is sorted 
  in ascending order */
template<typename T, template <typename> class ARMA_VECTOR_TYPE>
uword binarySearch(const ARMA_VECTOR_TYPE<T> &v, const uword target){
  uword start = 0;
  uword end = v.n_rows - 1;
  while (true){
    uword mid = (start + end) / 2;
    if (end <= start) return mid;
    if (mid > 0 && v[mid - 1] < target && v[mid] >= target)
      return mid;        
    if (v[mid] < target)
      start = mid + 1;
    else
      end = mid - 1;
  }
}