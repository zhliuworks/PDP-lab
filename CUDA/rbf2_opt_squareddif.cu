/*
 *  RBF kernel computation -- optimize squared difference (2), based on reduction optimization (1)
 *
 *  Reference: Better Performance at Lower Occupancy
 *  available at: https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf
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
#define REPEATED_TIMES 500000
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
    res = 0;
    for (int i = 0; i < size; i++) {
        tmp = matA[i] - matB[i];
        res += tmp * tmp;
    }
    return res;
}


__global__ void squared_diff_0(int *matAd, int *matBd, int *matCd) {
    /* baseline: matAd, matBd, matCd are all accessed from global memory */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = matAd[idx] - matBd[idx];
    matCd[idx] = tmp * tmp;
}


__global__ void squared_diff_1(int *matId, int *matOd, int size) {
    /* optimized_1: input matrices are accessed from contiguous global memory, 
    (matId: 2*size, matA...matB...), for better locality */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = matId[idx] - matId[idx + size];
    matOd[idx] = tmp * tmp;
}


__global__ void squared_diff_2(int *matId, int *matOd, int size) {
    /* optimized_: cache-conscious data layout, (matId: 2*size,
    matA[0]-matB[0]-matA[1]-matB[1]..., for better locality) */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = matId[idx << 1] - matId[(idx << 1) + 1];
    matOd[idx] = tmp * tmp;
}


__global__ void squared_diff_3(int *matId, int *matOd, int size) {
    /* optimized_2: 2x more work per thread, for exploiting ILP */
    int idx = (blockIdx.x * blockDim.x << 1) + threadIdx.x;
    int tmp[2] = {0, 0};  // array is allocated in registers
    tmp[0] = matId[idx << 1] - matId[(idx << 1) + 1];
    tmp[1] = matId[(idx << 1) + blockDim.x] - matId[(idx << 1) + 1 + blockDim.x];
    matOd[idx] = tmp[0] * tmp[0];
    matOd[idx + blockDim.x] = tmp[1] * tmp[1];
}


__global__ void squared_diff_4(int *matId, int *matOd, int size) {
    /* optimized_4: 4x more work per thread, for exploiting ILP */
    int idx = (blockIdx.x * blockDim.x << 2) + threadIdx.x;
    int tmp[4] = {0, 0, 0, 0}; 
    tmp[0] = matId[idx << 1] - matId[(idx << 1) + 1];
    tmp[1] = matId[(idx << 1) + blockDim.x] - matId[(idx << 1) + 1 + blockDim.x];
    tmp[2] = matId[(idx << 1) + (blockDim.x << 1)] - matId[(idx << 1) + 1 + (blockDim.x << 1)];
    tmp[3] = matId[(idx << 1) + (blockDim.x << 1) + blockDim.x] - matId[(idx << 1) + 1 + (blockDim.x << 1) + blockDim.x];
    matOd[idx] = tmp[0] * tmp[0];
    matOd[idx + blockDim.x] = tmp[1] * tmp[1];
    matOd[idx + (blockDim.x << 1)] = tmp[2] * tmp[2];
    matOd[idx + (blockDim.x << 1) + blockDim.x] = tmp[3] * tmp[3];
}


__device__ void warpReduce(volatile int *matS, int tx) {
    // unroll the last warp
    matS[tx] += matS[tx + 32];
    matS[tx] += matS[tx + 16];
    matS[tx] += matS[tx + 8];
    matS[tx] += matS[tx + 4];
    matS[tx] += matS[tx + 2];
    matS[tx] += matS[tx + 1];
}


__global__ void reduction(int *matIn, int *matOut, int matIn_size) {
    /* reduction final optimized: unroll the last warp */
    extern __shared__ int matS[];

    int tx = threadIdx.x;
    int idx = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    matS[tx] = (idx < matIn_size) ? matIn[idx]: 0;
    matS[tx] += (idx + blockDim.x < matIn_size) ? matIn[idx + blockDim.x] : 0;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1) {
        if (tx < stride) {
            matS[tx] += matS[tx + stride];
        }
        __syncthreads();
    }

    if (tx < 32) {
        warpReduce(matS, tx);
    }

    if (tx == 0) {
        matOut[blockIdx.x] = matS[0];
    }    
}


__host__ int RBF_Compute_GPU_0(int *matA, int *matB, int size) {
    // compute RBF kernel on GPU, serve as a wrapper function
    int *matAd, *matBd, *matId, *matOd;
    int res;
    clock_t start, end;

    CUDA_CHECK(cudaMalloc((void**)&matAd, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matBd, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matId, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matOd, sizeof(int) * (size / NUM_THREADS)));

    CUDA_CHECK(cudaMemcpy(matAd, matA, sizeof(int) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matBd, matB, sizeof(int) * size, cudaMemcpyHostToDevice));
    
    dim3 dimGrid(size / NUM_THREADS, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    start = clock();
    for (int n = 0; n < REPEATED_TIMES; n++) {
        // baseline
        squared_diff_0<<<dimGrid, dimBlock>>>(matAd, matBd, matId);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        dim3 dimGrid((size / NUM_THREADS) >> 1, 1, 1);
        reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        int curr_size = (size / NUM_THREADS) >> 1;
        while (curr_size > 1) {
            CUDA_CHECK(cudaMemcpy(matId, matOd, sizeof(int) * curr_size, cudaMemcpyDeviceToDevice));
            dim3 dimGrid(curr_size / NUM_THREADS + 1, 1, 1);
            reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, curr_size);
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            curr_size /= NUM_THREADS;
        }
        CUDA_CHECK(cudaMemcpy(&res, matOd, sizeof(int), cudaMemcpyDeviceToHost));
    }
    end = clock();
    printf("GPU (base) elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    CUDA_CHECK(cudaFree(matAd));
    CUDA_CHECK(cudaFree(matBd));
    CUDA_CHECK(cudaFree(matId));
    CUDA_CHECK(cudaFree(matOd));

    return res;
}


__host__ int RBF_Compute_GPU_1(int *matA, int *matB, int size) {
    // compute RBF kernel on GPU, serve as a wrapper function
    int *matAd, *matId, *matOd;
    int res;
    clock_t start, end;

    CUDA_CHECK(cudaMalloc((void**)&matAd, sizeof(int) * size << 1));
    CUDA_CHECK(cudaMalloc((void**)&matId, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matOd, sizeof(int) * (size / NUM_THREADS)));

    CUDA_CHECK(cudaMemcpy(matAd, matA, sizeof(int) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matAd + size, matB, sizeof(int) * size, cudaMemcpyHostToDevice));

    dim3 dimGrid(size / NUM_THREADS, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    start = clock();
    for (int n = 0; n < REPEATED_TIMES; n++) {
        // input matrices are stored in contiguous memory
        squared_diff_1<<<dimGrid, dimBlock>>>(matAd, matId, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        dim3 dimGrid((size / NUM_THREADS) >> 1, 1, 1);
        reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        int curr_size = (size / NUM_THREADS) >> 1;
        while (curr_size > 1) {
            CUDA_CHECK(cudaMemcpy(matId, matOd, sizeof(int) * curr_size, cudaMemcpyDeviceToDevice));
            dim3 dimGrid(curr_size / NUM_THREADS + 1, 1, 1);
            reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, curr_size);
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            curr_size /= NUM_THREADS;
        }
        CUDA_CHECK(cudaMemcpy(&res, matOd, sizeof(int), cudaMemcpyDeviceToHost));
    }
    end = clock();
    printf("GPU (opt1) elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    CUDA_CHECK(cudaFree(matAd));
    CUDA_CHECK(cudaFree(matId));
    CUDA_CHECK(cudaFree(matOd));

    return res;
}


__host__ int RBF_Compute_GPU_2(int *matA, int *matB, int size) {
    // compute RBF kernel on GPU, serve as a wrapper function
    int *matAd, *matId, *matOd;
    int res;
    clock_t start, end;

    CUDA_CHECK(cudaMalloc((void**)&matAd, sizeof(int) * size << 1));
    CUDA_CHECK(cudaMalloc((void**)&matId, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matOd, sizeof(int) * (size / NUM_THREADS)));

    for (int i = 0; i < size; i++) {
        CUDA_CHECK(cudaMemcpy(matAd + (i << 1), matA + i, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matAd + (i << 1) + 1, matB + i, sizeof(int), cudaMemcpyHostToDevice));
    }

    dim3 dimGrid(size / NUM_THREADS, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    start = clock();
    for (int n = 0; n < REPEATED_TIMES; n++) {
        // allowing for cache-conscious layout
        squared_diff_2<<<dimGrid, dimBlock>>>(matAd, matId, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        dim3 dimGrid((size / NUM_THREADS) >> 1, 1, 1);
        reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        int curr_size = (size / NUM_THREADS) >> 1;
        while (curr_size > 1) {
            CUDA_CHECK(cudaMemcpy(matId, matOd, sizeof(int) * curr_size, cudaMemcpyDeviceToDevice));
            dim3 dimGrid(curr_size / NUM_THREADS + 1, 1, 1);
            reduction<<<dimGrid, dimBlock, SHARED_MEM_SIZE>>>(matId, matOd, curr_size);
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            curr_size /= NUM_THREADS;
        }
        CUDA_CHECK(cudaMemcpy(&res, matOd, sizeof(int), cudaMemcpyDeviceToHost));
    }
    end = clock();
    printf("GPU (opt2) elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    CUDA_CHECK(cudaFree(matAd));
    CUDA_CHECK(cudaFree(matId));
    CUDA_CHECK(cudaFree(matOd));

    return res;
}


__host__ int RBF_Compute_GPU_3(int *matA, int *matB, int size) {
    // compute RBF kernel on GPU, serve as a wrapper function
    int *matAd, *matId, *matOd;
    int res;
    clock_t start, end;

    CUDA_CHECK(cudaMalloc((void**)&matAd, sizeof(int) * size << 1));
    CUDA_CHECK(cudaMalloc((void**)&matId, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matOd, sizeof(int) * (size / NUM_THREADS)));

    for (int i = 0; i < size; i++) {
        CUDA_CHECK(cudaMemcpy(matAd + (i << 1), matA + i, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matAd + (i << 1) + 1, matB + i, sizeof(int), cudaMemcpyHostToDevice));
    }

    start = clock();
    for (int n = 0; n < REPEATED_TIMES; n++) {
        // 2x fewer threads, 2x more per-thread work
        dim3 dimGrid1(size / NUM_THREADS, 1, 1);
        dim3 dimBlock1(NUM_THREADS >> 1, 1, 1);  // 2x fewer
        squared_diff_3<<<dimGrid1, dimBlock1>>>(matAd, matId, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        dim3 dimGrid2((size / NUM_THREADS) >> 1, 1, 1);
        dim3 dimBlock2(NUM_THREADS, 1, 1);
        reduction<<<dimGrid2, dimBlock2, SHARED_MEM_SIZE>>>(matId, matOd, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        int curr_size = (size / NUM_THREADS) >> 1;
        while (curr_size > 1) {
            CUDA_CHECK(cudaMemcpy(matId, matOd, sizeof(int) * curr_size, cudaMemcpyDeviceToDevice));
            dim3 dimGrid(curr_size / NUM_THREADS + 1, 1, 1);
            reduction<<<dimGrid2, dimBlock2, SHARED_MEM_SIZE>>>(matId, matOd, curr_size);
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            curr_size /= NUM_THREADS;
        }
        CUDA_CHECK(cudaMemcpy(&res, matOd, sizeof(int), cudaMemcpyDeviceToHost));
    }
    end = clock();
    printf("GPU (opt3) elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    CUDA_CHECK(cudaFree(matAd));
    CUDA_CHECK(cudaFree(matId));
    CUDA_CHECK(cudaFree(matOd));

    return res;
}


__host__ int RBF_Compute_GPU_4(int *matA, int *matB, int size) {
    // compute RBF kernel on GPU, serve as a wrapper function
    int *matAd, *matId, *matOd;
    int res;
    clock_t start, end;

    CUDA_CHECK(cudaMalloc((void**)&matAd, sizeof(int) * size << 1));
    CUDA_CHECK(cudaMalloc((void**)&matId, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc((void**)&matOd, sizeof(int) * (size / NUM_THREADS)));

    for (int i = 0; i < size; i++) {
        CUDA_CHECK(cudaMemcpy(matAd + (i << 1), matA + i, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matAd + (i << 1) + 1, matB + i, sizeof(int), cudaMemcpyHostToDevice));
    }

    start = clock();
    for (int n = 0; n < REPEATED_TIMES; n++) {
        // 4x fewer threads, 4x more per-thread work
        dim3 dimGrid1(size / NUM_THREADS, 1, 1);
        dim3 dimBlock1(NUM_THREADS >> 2, 1, 1);  // 4x fewer
        squared_diff_4<<<dimGrid1, dimBlock1>>>(matAd, matId, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        dim3 dimGrid2((size / NUM_THREADS) >> 1, 1, 1);
        dim3 dimBlock2(NUM_THREADS, 1, 1);
        reduction<<<dimGrid2, dimBlock2, SHARED_MEM_SIZE>>>(matId, matOd, size);
        cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());

        int curr_size = (size / NUM_THREADS) >> 1;
        while (curr_size > 1) {
            CUDA_CHECK(cudaMemcpy(matId, matOd, sizeof(int) * curr_size, cudaMemcpyDeviceToDevice));
            dim3 dimGrid(curr_size / NUM_THREADS + 1, 1, 1);
            reduction<<<dimGrid2, dimBlock2, SHARED_MEM_SIZE>>>(matId, matOd, curr_size);
            cudaDeviceSynchronize(); CUDA_CHECK(cudaGetLastError());
            curr_size /= NUM_THREADS;
        }
        CUDA_CHECK(cudaMemcpy(&res, matOd, sizeof(int), cudaMemcpyDeviceToHost));
    }
    end = clock();
    printf("GPU (opt4) elapsed time: %f ms\n", (double)(end - start) / (CLOCKS_PER_SEC * REPEATED_TIMES * 0.001));

    CUDA_CHECK(cudaFree(matAd));
    CUDA_CHECK(cudaFree(matId));
    CUDA_CHECK(cudaFree(matOd));

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
        int res_gpu_0 = RBF_Compute_GPU_0(matA, matB, size);
        int res_gpu_1 = RBF_Compute_GPU_1(matA, matB, size);
        int res_gpu_2 = RBF_Compute_GPU_2(matA, matB, size);
        int res_gpu_3 = RBF_Compute_GPU_3(matA, matB, size);
        int res_gpu_4 = RBF_Compute_GPU_4(matA, matB, size);
        assert(res_cpu == res_gpu_0);
        assert(res_cpu == res_gpu_1);
        assert(res_cpu == res_gpu_2);
        assert(res_cpu == res_gpu_2);
        assert(res_cpu == res_gpu_3);
        assert(res_cpu == res_gpu_4);
    }

    free(matA); free(matB);
    return 0;
}