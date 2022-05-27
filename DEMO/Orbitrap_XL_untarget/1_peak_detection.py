import os
"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import sys

tmp_path = os.path.abspath(__file__)
root_path = '/'.join(tmp_path.split('/')[:-3])
sys.path.append(root_path)

from workflow.predict.main_eval import MsNetEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
network_dir = 'msnet_20220215_143158'
epoch = 300
data_root = os.path.join(root_path, 'dataset', 'Orbitrap_XL')
file_names = ['130124_dilA_1_01', '130124_dilA_1_02', '130124_dilA_1_03', '130124_dilA_1_04',
              '130124_dilA_2_01', '130124_dilA_2_02', '130124_dilA_2_03', '130124_dilA_2_04',
              '130124_dilA_2_05', '130124_dilA_2_06', '130124_dilA_2_07',
              '130124_dilA_3_01', '130124_dilA_3_02', '130124_dilA_3_03', '130124_dilA_3_04',
              '130124_dilA_3_05', '130124_dilA_3_06', '130124_dilA_3_07',
              '130124_dilA_4_01', '130124_dilA_4_02', '130124_dilA_4_03', '130124_dilA_4_04',
              '130124_dilA_4_05', '130124_dilA_4_06', '130124_dilA_4_07',
              '130124_dilA_5_01', '130124_dilA_5_02', '130124_dilA_5_03', '130124_dilA_5_04',
              '130124_dilA_6_01', '130124_dilA_6_02', '130124_dilA_6_03', '130124_dilA_6_04',
              '130124_dilA_7_01', '130124_dilA_7_02', '130124_dilA_7_03', '130124_dilA_7_04',
              '130124_dilA_8_01', '130124_dilA_8_02', '130124_dilA_8_03', '130124_dilA_8_04',
              '130124_dilA_9_01', '130124_dilA_9_02', '130124_dilA_9_03', '130124_dilA_9_04',
              '130124_dilA_10_01', '130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04',
              '130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04',
              '130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04']
data_dir = [os.path.join(data_root, 'Untarget-' + file_name) for file_name in file_names]

evaluator = MsNetEvaluator(exp=network_dir, epoch=epoch)
for eval_dir in data_dir:
    evaluator.eval(eval_dir=eval_dir, mass_analyzer='orbitrap', mz_resolution=60000, resolution_mz=400,
                   rt_fwhm=0.25, block_rt_width=6, block_mz_width=0.8, target_id=None)
