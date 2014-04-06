#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    size_t M = 10000;
    size_t K = 10000;
    size_t N = 10000;

    double *matA, *matB, *matC;
    matA = (double *) malloc(8 * 10000 * 10000);
    matB = (double *) malloc(8 * 10000 * 10000);
    matC = (double *) malloc(8 * 10000 * 10000);

    if (matA == NULL || matB == NULL || matC == NULL) return 1;

    printf("HERE!\n");

    clock_t t;
    t = clock();

    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            matC[i * N + j] = 0;
            for(int k=0;k<N;k++){
                matC[i * N + j] += matA[i * N + k] * matB[k * N + j];
            }
        }
    }

    printf("HERE 2!\n");

    t = clock() - t;
    printf("Took: %f seconds.\n", ((float)t)/CLOCKS_PER_SEC);
    return 0;
}