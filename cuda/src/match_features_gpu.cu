#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "match_features_gpu.h"
#include "cuda_utils.h"

__global__ void match_features_kernel_fast(int n, int s, float rt_tolerance, float mz_tolerance,
                                                const float *__restrict__ rt, const float *__restrict__ mz,
                                                int *__restrict__ match_status, int *__restrict__ match_position) {

    int base_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_col_idx = blockIdx.y;
    int col_idx = blockIdx.z;
    if (base_row_idx >= n || base_col_idx >= s || col_idx >= s) return;

    float tmp_mz = mz[base_row_idx * s + base_col_idx];
    float tmp_rt = rt[base_row_idx * s + base_col_idx];
    int tmp_match_status = 0;

    for (int p = 0; p < n; p++) {
        float target_mz = mz[p * s + col_idx];
        float target_rt = rt[p * s + col_idx];
        if (target_mz < tmp_mz - mz_tolerance) {
            continue;
        }
        if (target_mz > tmp_mz + mz_tolerance) {
            break;
        }
        if ((target_rt < tmp_rt - rt_tolerance) || (target_rt > tmp_rt + rt_tolerance)) {
            continue;
        }
        tmp_match_status = 1;
        match_position[base_col_idx * n * s + p * s + col_idx] = 1;
    }

    match_status[base_row_idx * s * s + base_col_idx * s + col_idx] = tmp_match_status;
}


void match_features_kernel_launcher_fast(int n, int s, float rt_tolerance, float mz_tolerance,
                                         const float *rt, const float *mz,
                                         int *match_status, int *match_position, cudaStream_t stream) {
    // rt: (N, S)
    // mz: (N, S)
    // output:
    //      match_status: (N, S, S)
    //      match_status: (S, N, S)

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), s, s);
    dim3 threads(THREADS_PER_BLOCK);

    match_features_kernel_fast<<<blocks, threads, 0, stream>>>(n, s, rt_tolerance, mz_tolerance, rt, mz, match_status, match_position);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}