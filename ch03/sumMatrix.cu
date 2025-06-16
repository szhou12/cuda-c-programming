#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

void initialData(float* ip, const int size) {
    int i;

    for (i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny) {
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("*** Arrays do not match. ***\n\n");
            break;
        }
    }
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float* A, float* B, float* C, int NX, int NY) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    printf("sumMatrix program starts ...");

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // init data at host side
    iStart = seconds();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    iElaps = seconds() - iStart;
    printf("Initialize Matrices on Host: %f ms\n", iElaps * 1000);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    printf("Sum Matrices on Host: %f ms\n", iElaps * 1000);

    // malloc device global memory
    float* d_MatA, * d_MatB, * d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;

    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // execute the kernel
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    sumMatrixOnGPU2D <<<grid, block>>> (d_MatA, d_MatB, d_MatC, nx, ny);
    iElaps = seconds() - iStart;
    printf("Sum Matrices on GPU: %f ms\n", iElaps * 1000);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    iStart = seconds();
    checkResult(hostRef, gpuRef, nxy);
    iElaps = seconds() - iStart;
    printf("Check result on Host: %f ms\n", iElaps * 1000);

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;






}