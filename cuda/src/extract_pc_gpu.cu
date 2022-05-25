#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "extract_pc_gpu.h"
#include "cuda_utils.h"


__global__ void extract_pc_kernel_fast(int n, int rt_len, int mz_len, float min_z, float rt_tolerance, float mz_tolerance,
    const float *__restrict__ target_rt, const float *__restrict__ target_mz,
    const float *__restrict__ xyz, int *__restrict__ idx) {

    int rt_idx = blockIdx.y;
    int mz_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mz_idx >= mz_len || rt_idx >= rt_len) return;

    float center_rt = target_rt[rt_idx];
    float center_mz = target_mz[mz_idx];

    float rt_start = center_rt - rt_tolerance;
    float rt_end = center_rt + rt_tolerance;
    float mz_start = center_mz - mz_tolerance;
    float mz_end = center_mz + mz_tolerance;

    int group = rt_idx % 2 + (mz_idx % 2) * 2;
    idx += group * n;

    int blk_idx = mz_idx * rt_len + rt_idx;

    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3];
        if (x <= rt_start) continue;
        if (x >= rt_end) break;

        float y = xyz[k * 3 + 1];
        if (y <= mz_start || y >= mz_end) continue;

        float z = xyz[k * 3 + 2];
        if (z < min_z) continue;

        idx[k] = blk_idx;
    }
}


void extract_pc_kernel_launcher_fast(int n, int rt_len, int mz_len, float min_z, float rt_tolerance, float mz_tolerance,
        const float *target_rt, const float *target_mz, const float *xyz, int *idx, cudaStream_t stream) {

    cudaError_t err;

    dim3 blocks(DIVUP(mz_len, THREADS_PER_BLOCK), rt_len);
    dim3 threads(THREADS_PER_BLOCK);

    extract_pc_kernel_fast<<<blocks, threads, 0, stream>>>(n, rt_len, mz_len, min_z, rt_tolerance, mz_tolerance, target_rt, target_mz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}