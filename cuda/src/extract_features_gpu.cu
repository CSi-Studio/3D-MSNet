#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "extract_features_gpu.h"
#include "cuda_utils.h"


__global__ void extract_features_kernel_fast(int b, int n, int n_grid, float cell_size_x, float cell_size_y,
                                                const float *__restrict__ xyz, float *__restrict__ feature) {
    // xyz: (B, N, 3)
    // output:
    //      feature: (B, N, n_grid * n_grid)

    int n_cell = n_grid * n_grid;
    int bs_idx = blockIdx.y;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = thread_idx / n_cell;
    if (bs_idx >= b || pt_idx >= n) return;
    int cell_idx = thread_idx % n_cell;
    int cell_row = cell_idx / n_grid;
    int cell_col = cell_idx % n_grid;

    xyz += bs_idx * n * 3;
    const float* center_xyz = xyz + pt_idx * 3;
    feature += bs_idx * n * n_cell + pt_idx * n_cell + cell_idx;

    float lower_x = center_xyz[0] - n_grid * cell_size_x / 2.0f + cell_size_x * cell_col;
    float lower_y = center_xyz[1] - n_grid * cell_size_y / 2.0f + cell_size_y * cell_row;
    float higher_x = lower_x + cell_size_x;
    float higher_y = lower_y + cell_size_y;

    float intensity[1000];
    float total_intensity = 0;
    int match_count = 0;
    for (int p = 0; p < n; ++p) {
        float x = xyz[p * 3 + 0];
        float y = xyz[p * 3 + 1];
        float z = xyz[p * 3 + 2];
        if (x <= lower_x || x >= higher_x || y <= lower_y || y >= higher_y) {
            continue;
        }
        intensity[match_count] = z;
        total_intensity += z;
        match_count ++;
        if (match_count == 1000) {
            break;
        }
    }
    if (match_count == 0) {
        feature[0] = 0;
    } else {
        // feature[0] = intensity[match_count / 2];
        // feature[0] = intensity[0];
        feature[0] = total_intensity / match_count;
        // feature[0] = match_count;
    }
}


void extract_features_kernel_launcher_fast(int b, int n, int n_grid, float cell_size_x, float cell_size_y, const float *xyz,
                                            float *feature, cudaStream_t stream) {
    // xyz: (B, N, 3)
    // output:
    //      feature: (B, N, n_grid * n_grid)

    cudaError_t err;

    dim3 blocks(DIVUP(n * n_grid * n_grid, THREADS_PER_BLOCK), b);
    dim3 threads(THREADS_PER_BLOCK);

    extract_features_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, n_grid, cell_size_x, cell_size_y, xyz, feature);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}