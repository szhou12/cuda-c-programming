#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../common/common.h"

/**
 * $ nvcc -arch=sm_61 -rdc=true nestedHelloWorld.cu -o nestedHelloWorld -lcudadevrt
 * $ ./nestedHelloWorld
 * $ ./nestedHelloWorld 2
 */
__global__ void nestedHelloWorld(int const iSize,int iDepth) {
    int tid = threadIdx.x;

    printf("Recursion=%d: Hello World from thread %d of block %d\n", iDepth, tid, blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1) return;

    // reduce block size to half
    int nthreads = iSize>>1;

    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0) {
        // launch child grid with half the threads
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char** argv) {
    int size = 8;
    int blocksize = 8;
    int igrid = 1;

    if (argc > 1) {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x, block.x);

    nestedHelloWorld << <grid, block >> > (block.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}