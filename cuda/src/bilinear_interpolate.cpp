#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "bilinear_interpolate_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int bilinear_neighbor_wrapper_fast(int b, int n, int m, int grid_x, int grid_y, float l_x, float l_y,
                                   at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                                   at::Tensor idx_tensor, at::Tensor weight_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(new_xyz_tensor);
    const float *xyz = xyz_tensor.data<float>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    float *weight = weight_tensor.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bilinear_neighbor_kernel_launcher_fast(b, n, m, grid_x, grid_y, l_x, l_y, xyz, new_xyz, idx, weight, stream);
    return 1;
}

int bilinear_interpolate_wrapper_fast(int b, int n, int m, int c, int k,
                                      at::Tensor feature_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor,
                                      at::Tensor new_feature_tensor) {
    CHECK_INPUT(feature_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(weight_tensor);
    const float *feature = feature_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const float *weight = weight_tensor.data<float>();
    float *new_feature = new_feature_tensor.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bilinear_interpolate_kernel_launcher_fast(b, n, m, c, k, feature, idx, weight, new_feature, stream);
    return 1;
}

int bilinear_interpolate_grad_wrapper_fast(int b, int n, int m, int c, int k,
                                           at::Tensor grad_out_tensor, at::Tensor idx_tensor,
                                           at::Tensor weight_tensor, at::Tensor grad_point_tensor) {
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(weight_tensor);
    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const float *weight = weight_tensor.data<float>();
    float *grad_point = grad_point_tensor.data<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bilinear_interpolate_grad_kernel_launcher_fast(b, n, m, c, k, grad_out, idx, weight, grad_point, stream);
    return 1;
}