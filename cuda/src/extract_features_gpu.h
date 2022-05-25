#ifndef _EXTRACT_FEATURES_GPU_H
#define _EXTRACT_FEATURES_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int extract_features_wrapper_fast(int b, int n, int n_grid, float cell_size_x, float cell_size_y, at::Tensor xyz_tensor,
                                    at::Tensor feature_tensor);

void extract_features_kernel_launcher_fast(int b, int n, int n_grid, float cell_size_x, float cell_size_y, const float *xyz,
                                            float *feature, cudaStream_t stream);
#endif
