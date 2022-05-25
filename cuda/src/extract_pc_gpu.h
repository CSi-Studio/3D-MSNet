#ifndef _EXTRACT_PC_GPU_H
#define _EXTRACT_PC_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int extract_pc_wrapper_fast(int n, int rt_len, int mz_len, float min_z, float rt_tolerance, float mz_tolerance,
    at::Tensor target_rt, at::Tensor target_mz, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void extract_pc_kernel_launcher_fast(int n, int rt_len, int mz_len, float min_z, float rt_tolerance, float mz_tolerance,
    const float *target_rt, const float *target_mz, const float *xyz, int *idx, cudaStream_t stream);

#endif
