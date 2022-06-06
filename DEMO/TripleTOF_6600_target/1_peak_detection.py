"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import os
import sys

tmp_path = os.path.abspath(__file__)
root_path = '/'.join(tmp_path.split('/')[:-3])
sys.path.append(root_path)

from workflow.predict.main_eval import MsNetEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
network_dir = 'msnet_20220427_141044'
epoch = 1000
data_root = os.path.join(root_path, 'dataset', 'TripleTOF_6600')
data_dir = [os.path.join(data_root, 'Target-20170326-960MIX_SampleA_1'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleA_2'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleA_3'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleA_4'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleB_1'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleB_2'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleB_3'),
            os.path.join(data_root, 'Target-20170326-960MIX_SampleB_4')]

evaluator = MsNetEvaluator(exp=network_dir, epoch=epoch)
for eval_dir in data_dir:
    evaluator.eval(eval_dir=eval_dir, mass_analyzer='tof', mz_resolution=35000, resolution_mz=956, rt_fwhm=0.1,
                   center_threshold=0.5, block_rt_width=6, block_mz_width=0.8, target_id=-1)
