#include <armadillo>

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void vunion(ARMA_VECTOR_TYPE<T> &first,
    ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
    std::vector<T> output;
    std::set_union(first.begin(), first.end(),
                    second.begin(), second.end(),
                    back_inserter(output)) ;
    result = arma::conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void vintersection(ARMA_VECTOR_TYPE<T> &first,
    ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
    std::vector<T> output;
    std::set_intersection(first.begin(), first.end(),
                        second.begin(), second.end(),
                        back_inserter(output)) ;
    result = arma::conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}

template<typename T, template <typename> class ARMA_VECTOR_TYPE>
void vdifference(ARMA_VECTOR_TYPE<T> &first,
    ARMA_VECTOR_TYPE<T> &second, ARMA_VECTOR_TYPE<T> &result) {
    std::vector<T> output;
    std::set_difference(first.begin(), first.end(),
                        second.begin(), second.end(),
                        back_inserter(output)) ;
    result = arma::conv_to<ARMA_VECTOR_TYPE<T>>::from(output);
}
