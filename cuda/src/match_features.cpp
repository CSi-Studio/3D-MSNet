#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "match_features_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int match_features_wrapper_fast(int n, int s, float rt_tolerance, float mz_tolerance,
                                at::Tensor rt_tensor, at::Tensor mz_tensor,
                                at::Tensor match_status_tensor, at::Tensor match_position_tensor) {
    CHECK_INPUT(mz_tensor);
    CHECK_INPUT(rt_tensor);
    const float *mz = mz_tensor.data<float>();
    const float *rt = rt_tensor.data<float>();
    int *match_status = match_status_tensor.data<int>();
    int *match_position = match_position_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    match_features_kernel_launcher_fast(n, s, rt_tolerance, mz_tolerance, rt, mz, match_status, match_position, stream);
    return 1;
}