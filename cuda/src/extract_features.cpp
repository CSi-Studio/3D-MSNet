#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "extract_features_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int extract_features_wrapper_fast(int b, int n, int n_grid, float cell_size_x, float cell_size_y,
                                    at::Tensor xyz_tensor, at::Tensor feature_tensor) {
    CHECK_INPUT(xyz_tensor);
    const float *xyz = xyz_tensor.data<float>();
    float *feature = feature_tensor.data<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    extract_features_kernel_launcher_fast(b, n, n_grid, cell_size_x, cell_size_y, xyz, feature, stream);
    return 1;
}