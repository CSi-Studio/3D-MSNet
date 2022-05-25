#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "bilinear_interpolate_gpu.h"
#include "cuda_utils.h"

__global__ void bilinear_neighbor_kernel_fast(int b, int n, int m, int grid_x, int grid_y, float l_x, float l_y,
                                                const float *__restrict__ xyz, const float *__restrict__ new_xyz,
                                                int *__restrict__ idx, float *__restrict__ weight) {
    // xyz: (B, N, 3)
    // new_xyz: (B, M, 3)
    // output:
    //      weight: (B, M, (2 * grid_x + 1) * (2 * grid_y + 1), 100)
    //      idx: (B, M, (2 * grid_x + 1) * (2 * grid_y + 1), 100)

    int n_kp = (2 * grid_x + 1) * (2 * grid_y + 1);
    int bs_idx = blockIdx.z;
    int kp_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || kp_idx >= n_kp || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    idx += bs_idx * m * n_kp * 100 + pt_idx * n_kp * 100 + kp_idx * 100;
    weight += bs_idx * m * n_kp * 100 + pt_idx * n_kp * 100 + kp_idx * 100;
    xyz += bs_idx * n * 3;

    int x_idx = kp_idx % (2 * grid_x + 1) - grid_x;
    int y_idx = grid_y - kp_idx / (2 * grid_x + 1);

    int cnt = 0;
    float total_weight = 0;
    for (int i = 0; i < n; i++) {
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float diff_x = abs(x - new_xyz[0] - x_idx * l_x);
        float diff_y = abs(y - new_xyz[1] - y_idx * l_y);

        if (diff_x >= l_x || diff_y >= l_y) {
            continue;
        }

        weight[cnt] = (l_x - diff_x) * (l_y - diff_y);
        idx[cnt] = i;
        total_weight += weight[cnt];

        cnt++;
        if (cnt >= 100) {
            break;
        }
    }

    for (int i = 0; i < 100; i ++) {
        if (idx[i] == -1) {
            break;
        }
        weight[i] /= total_weight;
    }

}

void bilinear_neighbor_kernel_launcher_fast(int b, int n, int m, int grid_x, int grid_y, float l_x, float l_y,
                                            const float *xyz, const float *new_xyz, int *idx, float *weight, cudaStream_t stream) {
    // xyz: (B, N, 3)
    // new_xyz: (B, M, 3)
    // output:
    //      weight: (B, M, (2 * grid_x + 1) * (2 * grid_y + 1), 100)
    //      idx: (B, M, (2 * grid_x + 1) * (2 * grid_y + 1), 100)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), (2 * grid_x + 1) * (2 * grid_y + 1), b);
    dim3 threads(THREADS_PER_BLOCK);

    bilinear_neighbor_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, grid_x, grid_y, l_x, l_y, xyz, new_xyz, idx, weight);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void bilinear_interpolate_kernel_fast(int b, int n, int m, int c, int k, const float *__restrict__ feature,
                                                const int *__restrict__ idx, const float *__restrict__ weight,
                                                float *__restrict__ new_feature) {
    // xyz: (B, N, 3)
    // feature: (B, N, C)
    // idx: (B, M, K, 100)
    // weight: (B, M, K, 100)
    // output:
    //      new_feature: (B, M, K, C)

    int bs_idx = blockIdx.z;
    int kp_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || kp_idx >= k || pt_idx >= m) return;

    new_feature += bs_idx * m * k * c + pt_idx * k * c + kp_idx * c;
    feature += bs_idx * n * c;
    idx += bs_idx * m * k * 100 + pt_idx * k * 100 + kp_idx * 100;
    weight += bs_idx * m * k * 100 + pt_idx * k * 100 + kp_idx * 100;

    for (int i = 0; i < 100; i ++) {
        if (idx[i] == -1) {
            break;
        }
        for (int j = 0; j < c; j ++) {
            new_feature[j] += weight[i] * feature[idx[i] * c + j];
        }
    }
}



void bilinear_interpolate_kernel_launcher_fast(int b, int n, int m, int c, int k,
                                                const float *feature, const int *idx, const float *weight,
                                                float *new_feature, cudaStream_t stream) {
    // xyz: (B, N, 3)
    // feature: (B, N, C)
    // idx: (B, M, K, 100)
    // weight: (B, M, K, 100)
    // output:
    //      new_feature: (B, M, K, C)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), k, b);
    dim3 threads(THREADS_PER_BLOCK);

    bilinear_interpolate_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, c, k, feature, idx, weight, new_feature);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void bilinear_interpolate_grad_kernel_fast(int b, int n, int m, int c, int k, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ grad_point) {
    // grad_out: (B, M, K, C)
    // idx: (B, M, K, 100)
    // weight: (B, M, K, 100)
    // output:
    //      grad_point: (B, N, C)

    int bs_idx = blockIdx.z;
    int kp_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || kp_idx >= k || pt_idx >= m) return;

    grad_out += bs_idx * m * k * c + pt_idx * k * c + kp_idx * c;
    idx += bs_idx * m * k * 100 + pt_idx * k * 100 + kp_idx * 100;
    weight += bs_idx * m * k * 100 + pt_idx * k * 100 + kp_idx * 100;
    grad_point += bs_idx * n * c;

    for (int i = 0; i < 100; i ++) {
        if (idx[i] == -1) {
            break;
        }
        for (int j = 0; j < c; j ++) {
            atomicAdd(grad_point + idx[i] * c + j, grad_out[j] * weight[i]);
        }
    }
}

void bilinear_interpolate_grad_kernel_launcher_fast(int b, int n, int m, int c, int k, const float *grad_out,
    const int *idx, const float *weight, float *grad_point, cudaStream_t stream) {
    // grad_out: (B, M, K, C)
    // idx: (B, M, K, 100)
    // weight: (B, M, K, 100)
    // output:
    //      grad_point: (B, N, C)

    cudaError_t err;
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), k, b);
    dim3 threads(THREADS_PER_BLOCK);
    bilinear_interpolate_grad_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, c, k, grad_out, idx, weight, grad_point);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}