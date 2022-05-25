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
import argparse

tmp_path = os.path.abspath(__file__)
root_path = '/'.join(tmp_path.split('/')[:-3])
sys.path.append(root_path)

from workflow.predict.point_cloud_extractor import extract


parser = argparse.ArgumentParser(description='PXD001091-raw-untarget data preparation')

parser.add_argument('--data_dir', type=str, help='converted file dir', default=os.path.join(root_path, 'dataset', 'PXD001091', 'mzml'))
parser.add_argument('--output_dir', type=str, help='point cloud output directory', default=os.path.join(root_path, 'dataset', 'PXD001091'))
parser.add_argument('--lib_path', type=str, help='library')
parser.add_argument('--mode', type=str, help='acquisition method', default='DDA')
parser.add_argument('--window_mz_width', type=float, help='window_mz_width', default=0.8)
parser.add_argument('--window_rt_width', type=float, help='window_rt_width', default=6)
parser.add_argument('--min_intensity', type=float, help='min_intensity', default=1000)
parser.add_argument('--from_mz', type=float, help='from_mz', default=400)
parser.add_argument('--to_mz', type=float, help='to_mz', default=2000)
parser.add_argument('--from_rt', type=float, help='from_rt', default=0)
parser.add_argument('--to_rt', type=float, help='to_rt', default=120)
parser.add_argument('--max_peak_mz_width', type=float, help='max_peak_mz_width', default=0.2)
parser.add_argument('--max_peak_rt_width', type=float, help='max_peak_rt_width', default=2)
args = parser.parse_args()

extract(args)
