#ifndef _MS_QUERY_GPU_H
#define _MS_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ms_query_wrapper_fast(int b, int n, int m, float radius, int nsample,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ms_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
	const float *xyz, const float *new_xyz, int *idx, cudaStream_t stream);

#endif
