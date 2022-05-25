"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='msnet',
    ext_modules=[
        CUDAExtension('msnet_cuda', [
            'src/msnet_api.cpp',

            'src/bilinear_interpolate.cpp',
            'src/bilinear_interpolate_gpu.cu',

            'src/extract_features.cpp',
            'src/extract_features_gpu.cu',

            'src/match_features.cpp',
            'src/match_features_gpu.cu',

            'src/extract_pc.cpp',
            'src/extract_pc_gpu.cu',

            'src/ms_query.cpp',
            'src/ms_query_gpu.cu',

            'src/ball_query.cpp',
            'src/ball_query_gpu.cu',
            'src/ball_query2.cpp', 
            'src/ball_query2_gpu.cu',
            'src/group_points.cpp', 
            'src/group_points_gpu.cu',
            'src/interpolate.cpp', 
            'src/interpolate_gpu.cu',
            'src/sampling.cpp', 
            'src/sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
