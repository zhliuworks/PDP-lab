/*
 *  RBF kernel computation -- baseline (0)
 *
 *  Note that:
 *  1. There is no intrinsic difference between matrix 
 *  and vector in the computation of RBF kernel, so the matrix
 *  is seen as a 1D vector for simplicity.
 *  2. The RBF kernel is too small, so only the l2-norm is 
 *  computed, i.e. ||xi - xj||^2
 *
 *  Author: Zi-Han Liu
 */

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define SIZE_MIN 1024
#define SIZE_MAX 1048576
#define SIZE_STEP 4
#define REPEATED_TIMES 100000
#define NUM_THREADS 256
#define SHARED_MEM_SIZE sizeof(int) * NUM_THREADS
// #define SIGMA 0.5

// #define RBF_Final(l2_norm, sigma) \
//     exp(-1.0 * l2_norm / (2.0 * sigma * sigma))

#define CUDA_CHECK(call) { \
    cudaError_t stat = (call); \
    if (stat != cudaSuccess) { \
        printf("[CUDA ERROR %d] %s\n", stat, cudaGetErrorString(stat)); \
        cudaDeviceReset(); assert(false); \
    } \
}


__host__ int RBF_Compute_CPU(int *matA, int *matB, int size) {
    // compute RBF kernel on CPU

    register int res, tmp;
    clock_t start, end;

    start = clock();
    // repeat for REPEATED_TIMES and average the elapsed time
    for (int n = 0; n < REPEATED_TIMES; n++) {
        res = 0;
        for (int i = 0; i < size; i++) {
            tmp = matA[i] - matB[i];
            res += tmp * tmp;
        }
    }
    end = clock();
    printf("CPU elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    // return RBF_Final(res, SIGMA); // too small...
    return res;
}


__global__ void squared_diff(int *matAd, int *matBd, int *matCd) {
    /* baseline: matAd, matBd, matCd are all accessed from global memory */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = matAd[idx] - matBd[idx];
    matCd[idx] = tmp * tmp;
}


__global__ void reduction(int *matIn, int *matOut, int matIn_size) {
    /* baseline: interleaved addressing */
    // perform reduction within a thread block, processed in shared memory
    extern __shared__ int matS[];  // size: NUM_THREADS
    // each thread loads one element from global to shared memory
    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    matS[tx] = (idx < matIn_size) ? matIn[idx]: 0;
    __syncthreads();
    // do reduction in shared memory
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tx % (2 * stride) == 0) {
            matS[tx] += matS[tx + stride];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tx == 0) {
        matOut[blockIdx.x] = matS[0];
    }
}


__host__ int RBF_Compute_GPU(int *matA, int *matB, int size) {
    // compute RBF kernel on GPU, serve as a wrapper function

    // matAd, matBd are two input matrices (size)
    // matId is the squared difference of the two (size)
    // matOd is the reduction result (size/NUM_THREADS, per block)
    int *matAd, *matBd, *matId, *matOd;
    int res;
    clock_t start, end;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&matAd, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matBd, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matId, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matOd, sizeof(int) * (size / NUM_THREADS)));

    // copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(matAd, matA, sizeof(int) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matBd, matB, sizeof(int) * size, cudaMemcpyHostToDevice));

    // grid/block dimensions
    dim3 dimGrid(size / NUM_THREADS, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);
    
    start = clock();
    // repeat for REPEATED_TIMES and average the elapsed time
    for (int n = 0; n < REPEATED_TIMES; n++) {
        // compute squared difference of each position: matId[i] = (matAd[i] - matBd[i])^2
        squared_diff<<<dimGrid, dimBlock>>>(matAd, matBd, matId);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        // reduction within thread blocks
        reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        // reduction across thread blocks
        int curr_size = size / NUM_THREADS;
        while (curr_size > 1) {
            // matId is copied from matOd, and serves as the reduction input of next iter.
            CUDA_CHECK(cudaMemcpy(matId, matOd, sizeof(int) * curr_size, cudaMemcpyDeviceToDevice));
            // modify the number of blocks
            dim3 dimGrid(curr_size / NUM_THREADS + 1, 1, 1);
            // reduction again
            reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, curr_size);
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            curr_size /= NUM_THREADS;
        }
        CUDA_CHECK(cudaMemcpy(&res, matOd, sizeof(int), cudaMemcpyDeviceToHost));
    }
    end = clock();
    printf("GPU elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    // free device memory
    CUDA_CHECK(cudaFree(matAd));
    CUDA_CHECK(cudaFree(matBd));
    CUDA_CHECK(cudaFree(matId));
    CUDA_CHECK(cudaFree(matOd));

    // return RBF_Final(res, SIGMA); // too small...
    return res;
}


__host__ int main() {
    int *matA, *matB;
    matA = (int*)malloc(sizeof(int) * SIZE_MAX);
    matB = (int*)malloc(sizeof(int) * SIZE_MAX);
    
    printf("[INFO] initializing input matrices...\n");
    for (int i = 0; i < SIZE_MAX; i++) {
        matA[i] = 0;
        matB[i] = 1;
    }
    
    printf("[INFO] computing RBF kernel of the two matrices...\n");
    for (int size = SIZE_MIN; size <= SIZE_MAX; size *= SIZE_STEP) {
        printf("\n# Matrix size: %d\n", size);
        int res_cpu = RBF_Compute_CPU(matA, matB, size);
        int res_gpu = RBF_Compute_GPU(matA, matB, size);
        // printf("CPU result: %d\n", res_cpu);
        // printf("GPU result: %d\n", res_gpu);
        assert(res_cpu == res_gpu);
    }

    free(matA); free(matB);
    return 0;
}