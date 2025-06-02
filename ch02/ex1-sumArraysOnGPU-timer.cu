#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include "../common/common.h"

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++) {
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }        
}

// one thread hands i-th element's addition
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}



/**
 * This program adds two large vectors (arrays) of floats both on the CPU (host) and the GPU (device), 
 * and compares the results to check for correctness and performance.
 */
int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // CUDA Device Setup
    int dev = 0; // Picks CUDA device 0 (your first GPU).
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name); // Gets and prints device properties (name, etc).
    CHECK(cudaSetDevice(dev)); // Sets that device for CUDA calls.
    
    // Host Memory Allocation
    // set up date size of vectors
    int nElem = 1<<24; // 2^24 elements (~16 million)
    printf("Vector size %d\n", nElem);
    // malloc host memory
    size_t nBytes = nElem * sizeof(float); // array of floats with length = nElem
    // Allocates four large float arrays on the CPU:
    // h_A, h_B: the input vectors
    // hostRef: to store CPU (host) computation result
    // gpuRef: to store GPU (device) computation result
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = cpuSecond();
    initialData (h_A, nElem);
    initialData (h_B, nElem);
    iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    iStart = cpuSecond();
    sumArraysOnHost (h_A, h_B, hostRef, nElem); // hostRef = h_A + h_B
    iElaps = cpuSecond() - iStart;

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int iLen = 1023; // Exerice 1
    dim3 block (iLen); // 1024 threads per block - only one dimension .x
    dim3 grid ((nElem + block.x - 1)/block.x);
    // NOTE: total # of threads = nElem
    // i.e. one thread handles one element

    iStart = cpuSecond();
    sumArraysOnGPU <<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("sumArraysOnGPU <<<%d,%d>>> Time elapsed %f" \
        "sec\n", grid.x, block.x, iElaps);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return(0);

}