#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/common.h"

/*
 * This example demonstrates the impact of misaligned reads on performance by
 * forcing misaligned reads to occur on a float*. Kernels that reduce the
 * performance impact of misaligned reads via unrolling are also included below.
 */

void checkResult(float* hostRef, float* gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void initialData(float* ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}


void sumArraysOnHost(float* A, float* B, float* C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float* A, float* B, float* C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float* A, float* B, float* C, const int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffsetUnroll2(float* A, float* B, float* C, const int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int k = i + offset;

    if (k + blockDim.x < n)
    {
        C[i] = A[k] + B[k];
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

__global__ void readOffsetUnroll4(float* A, float* B, float* C, const int n,
    int offset)
{
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;

    if (k + 3 * blockDim.x < n)
    {
        C[i] = A[k] + B[k];
        C[i + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
        C[i + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}

int main(int argc, char** argv)
{
    std::cout << "readSegmentUnroll program starts ..." << std::endl;
    std::chrono::steady_clock::time_point begin;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size
    int nElem = 1 << 20; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset = atoi(argv[1]);

    if (argc > 2) blocksize = atoi(argv[2]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float* h_A = (float*)malloc(nBytes);
    float* h_B = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    //  summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    //  kernel 1:
    begin = StartTimer();
    warmup << <grid, block >> > (d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    std::cout << "warmup: " << GetDurationInMicroSeconds(begin, StopTimer()) << " microsecs" << std::endl;
    CHECK(cudaGetLastError());

    // kernel 1
    begin = StartTimer();
    readOffset << <grid, block >> > (d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    std::cout << "readOffset: " << GetDurationInMicroSeconds(begin, StopTimer()) << " microsecs" << std::endl;
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // kernel 2
    begin = StartTimer();
    readOffsetUnroll2 << <grid.x / 2, block >> > (d_A, d_B, d_C, nElem / 2, offset);
    CHECK(cudaDeviceSynchronize());
    std::cout << "readOffsetUnroll2: " << GetDurationInMicroSeconds(begin, StopTimer()) << " microsecs" << std::endl;
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // kernel 3
    begin = StartTimer();
    readOffsetUnroll4 << <grid.x / 4, block >> > (d_A, d_B, d_C, nElem / 4, offset);
    CHECK(cudaDeviceSynchronize());
    std::cout << "readOffsetUnroll4: " << GetDurationInMicroSeconds(begin, StopTimer()) << " microsecs" << std::endl;
    CHECK(cudaGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}