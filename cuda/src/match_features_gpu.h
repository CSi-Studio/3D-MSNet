#ifndef _MATCH_FEATURES_GPU_H
#define _MATCH_FEATURES_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int match_features_wrapper_fast(int n, int s, float rt_tolerance, float mz_tolerance,
                                  at::Tensor rt_tensor, at::Tensor mz_tensor,
                                  at::Tensor match_status_tensor, at::Tensor match_position_tensor);

void match_features_kernel_launcher_fast(int n, int s, float rt_tolerance, float mz_tolerance,
                                         const float *rt, const float *mz,
                                         int *match_status, int *match_position, cudaStream_t stream);
#endif
