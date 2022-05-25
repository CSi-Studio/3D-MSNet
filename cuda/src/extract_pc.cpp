#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "extract_pc_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int extract_pc_wrapper_fast(int n, int rt_len, int mz_len, float min_z, float rt_tolerance, float mz_tolerance,
    at::Tensor target_rt_tensor, at::Tensor target_mz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(target_rt_tensor);
    CHECK_INPUT(target_mz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *target_rt = target_rt_tensor.data<float>();
    const float *target_mz = target_mz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    extract_pc_kernel_launcher_fast(n, rt_len, mz_len, min_z, rt_tolerance, mz_tolerance, target_rt, target_mz, xyz, idx, stream);
    return 1;
}