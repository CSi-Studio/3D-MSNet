"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import numpy as np


def get_mz_fwhm(mz, mass_analyzer, resolution, resolution_mz):
    if mass_analyzer == 'tof':
        tmp_resolution = np.sqrt(mz / resolution_mz) * resolution
        return mz / tmp_resolution
    if mass_analyzer == 'orbitrap':
        tmp_resolution = resolution / np.sqrt(mz / resolution_mz)
        return mz / tmp_resolution
