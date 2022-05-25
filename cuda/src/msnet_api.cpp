#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "bilinear_interpolate_gpu.h"
#include "extract_features_gpu.h"
#include "match_features_gpu.h"
#include "extract_pc_gpu.h"
#include "ms_query_gpu.h"
#include "ball_query_gpu.h"
#include "ball_query2_gpu.h"
#include "group_points_gpu.h"
#include "sampling_gpu.h"
#include "interpolate_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilinear_neighbor_wrapper", &bilinear_neighbor_wrapper_fast, "bilinear_neighbor_wrapper_fast");
    m.def("bilinear_interpolate_wrapper", &bilinear_interpolate_wrapper_fast, "bilinear_interpolate_wrapper_fast");
    m.def("bilinear_interpolate_grad_wrapper", &bilinear_interpolate_grad_wrapper_fast, "bilinear_interpolate_grad_wrapper_fast");

    m.def("extract_features_wrapper", &extract_features_wrapper_fast, "extract_features_wrapper_fast");
    m.def("match_features_wrapper", &match_features_wrapper_fast, "match_features_wrapper_fast");
    m.def("extract_pc_wrapper", &extract_pc_wrapper_fast, "extract_pc_wrapper_fast");

    m.def("ms_query_wrapper", &ms_query_wrapper_fast, "ms_query_wrapper_fast");
    m.def("ball_query_wrapper", &ball_query_wrapper_fast, "ball_query_wrapper_fast");

    m.def("ball_query2_wrapper", &ball_query2_wrapper_fast, "ball_query2_wrapper_fast");

    m.def("group_points_wrapper", &group_points_wrapper_fast, "group_points_wrapper_fast");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast, "group_points_grad_wrapper_fast");

    m.def("gather_points_wrapper", &gather_points_wrapper_fast, "gather_points_wrapper_fast");
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast, "gather_points_grad_wrapper_fast");

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");
    
    m.def("three_nn_wrapper", &three_nn_wrapper_fast, "three_nn_wrapper_fast");
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast, "three_interpolate_wrapper_fast");
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast, "three_interpolate_grad_wrapper_fast");
}
