#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "../common/common.h"

#define LOG 0

// Recursive Implementation of Interleaved Pair Approach
int cpuRecursiveReduce(int *data, int const size) {
    // stop condition
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // recursive call
    return cpuRecursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
/**
 * Takes an input array g_idata, performs a sum reduction, and stores the result in g_odata.
 * Example: 4 threads per block
 *      Initial:    [1, 2, 3, 4, 5, 6, 7, 8]
 *                   ----------  ----------
 *                    block 0    block 1
 *      Step 1:     [3, 2, 7, 4, 11, 6, 15, 8]  (stride=1)
 *      Step 2:     [10, 2, 7, 4, 26, 6, 15, 8] (stride=2)
 *      Step 3:     [36, 2, 7, 4, 26, 6, 15, 8] (stride=4)
 * 
 */
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x; // thread id within this thread's block
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // global index of this thread across all blocks

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within thread block
        __syncthreads();
    }

    // write the result for this block to global memory
    it (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}

/**
 * 
 * Example: 
 *      Initial:    [1, 2, 3, 4, 5, 6, 7, 8]
 *      Step 1:     1st Call (isize = 8, 8 threads), istride = 8>>1 = 4
 *                  [6, 8, 10, 12, 5, 6, 7, 8]
 *      Step 2:     2nd Call (isize = 4, 4 threads), istride = 4>>1 = 2
 *                  [16, 20, 10, 12, 5, 6, 7, 8]
 *      Step 3:     3rd Call (isize = 2, 2 threads)
 *                  [36, 20, 10, 12, 5, 6, 7, 8]
 */
__global__ void gpuRecursiveReduce (int *g_idata, int *g_odata, unsigned int isize) {
    // set thread ID
    unsigned int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    // stop condition
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invocation: cut to half
    int istride = isize >> 1;

    if (istride > 1 && tid < istride) {
        // in-place reduction
        idata[tid] += idata[tid + istride];
    }

    // sync at block level
    __syncthreads();

    // nested invocation to generate child grids
    if (tid == 0) {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);

        // sync all child grids launched in this block
        cudaDeviceSynchronize();
    }

    // sync at block level again
    __syncthreads();
}

int main(int argc, char **argv) {
    // set up device
    int dev = 0, gpu_sum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // set up execution configuration
    int nblock  = 2048;
    int nthread = 512;   // initial block size

    if(argc > 1)
    {
        nblock = atoi(argv[1]);   // block size from command line argument
    }

    if(argc > 2)
    {
        nthread = atoi(argv[2]);   // block size from command line argument
    }

    int size = nblock * nthread; // total number of elements to reduceNeighbored

    dim3 block (nthread, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("array %d grid %d block %d\n", size, grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)( rand() & 0xFF );
        h_idata[i] = 1;
    }

    memcpy (tmp, h_idata, bytes);

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    double iStart, iElaps;

    // cpu recursive reduction
    iStart = seconds();
    int cpu_sum = cpuRecursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce\t\telapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // gpu reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored\t\telapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // gpu nested reduce kernel
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}