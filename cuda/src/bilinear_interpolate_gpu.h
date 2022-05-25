#ifndef _BILINEAR_INTERPOLATE_GPU_H
#define _BILINEAR_INTERPOLATE_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int bilinear_neighbor_wrapper_fast(int b, int n, int m, int grid_x, int grid_y, float l_x, float l_y,
                                    at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                                    at::Tensor idx_tensor, at::Tensor weight_tensor);

void bilinear_neighbor_kernel_launcher_fast(int b, int n, int m, int grid_x, int grid_y, float l_x, float l_y,
                                            const float *xyz, const float *new_xyz,
                                            int *idx, float *weight, cudaStream_t stream);

int bilinear_interpolate_wrapper_fast(int b, int n, int m, int c, int k,
                                      at::Tensor feature_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor,
                                      at::Tensor new_feature_tensor);

void bilinear_interpolate_kernel_launcher_fast(int b, int n, int m, int c, int k,
                                               const float *feature, const int *idx, const float *weight,
                                               float *new_feature, cudaStream_t stream);

int bilinear_interpolate_grad_wrapper_fast(int b, int n, int m, int c, int k,
                                           at::Tensor grad_out_tensor, at::Tensor idx_tensor,
                                           at::Tensor weight_tensor, at::Tensor grad_point_tensor);

void bilinear_interpolate_grad_kernel_launcher_fast(int b, int n, int m, int c, int k, const float *grad_out,
                                                    const int *idx, const float *weight,
                                                    float *grad_point, cudaStream_t stream);

#endif
