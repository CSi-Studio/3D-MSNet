#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "ms_query_gpu.h"
#include "cuda_utils.h"


__global__ void ms_query_kernel_fast(int b, int n, int m, float radius, int nsample,
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];

    int neighbors = 0;

    int neigh_idx[10000];
    float d2s[10000];

    for (int p = 0; p < n; ++p) {
        float x = xyz[p * 3 + 0];
        float y = xyz[p * 3 + 1];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y);
        if (d2 < radius2 && neighbors < 10000){
            neigh_idx[neighbors] = p;
            d2s[neighbors] = d2;
            ++neighbors;
        }
    }
    for (int s = 0; s < nsample; ++s) {
        int min_idx = 0;
        for (int k = 1; k < neighbors; ++k) {
            if (d2s[k] < d2s[min_idx]) {
                min_idx = k;
            }
        }
        if (d2s[min_idx] == 100) {
            for (; s < nsample; ++s) {
                idx[s] = idx[0];
            }
            break;
        }
        idx[s] = neigh_idx[min_idx];
        d2s[min_idx] = 100;
    }
}


void ms_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx, cudaStream_t stream) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ms_query_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}