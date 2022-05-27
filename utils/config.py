"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import argparse
import yaml
import os
import time

tmp_path = os.path.abspath(__file__)
root_path = '/'.join(tmp_path.split('/')[:-2])


def get_parser():
    args_cfg = argparse.FileType
    cfg_dir = os.path.join(os.path.dirname(__file__), "../config/msnet_default.yaml")
    with open(cfg_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join(root_path, 'experiment', cfg.model_name + '_' + str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))))
