"""
Copyright (c) 2020 CSi Biotech
3D-MSNet is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""

import torch
import time
import glob
import os
import csv
import sys
import argparse
import numpy as np

tmp_path = os.path.dirname(os.path.abspath(__file__))
root_path = '/'.join(tmp_path.split('/')[:-2])
sys.path.append(root_path)
from utils.config import cfg
from model.main_msnet import MsNet
from workflow.train.dataset_loader import fill_feature_cuda
from utils.polar_mask import get_point_instance, get_final_masks
from utils.ms_compatibility import get_mz_fwhm



class MsNetEvaluator:
    def __init__(self, exp, epoch):
        self.net = MsNet(cfg)
        self.net.backbone.load_state_dict(
            torch.load(root_path + '/%s/%s/%s_%.3d.pth' % ('experiment', exp, 'backbone', epoch)))
        self.net.sem_net.load_state_dict(
            torch.load(root_path + '/%s/%s/%s_%.3d.pth' % ('experiment', exp, 'sem_net', epoch)))
        self.net.center_net.load_state_dict(
            torch.load(root_path + '/%s/%s/%s_%.3d.pth' % ('experiment', exp, 'box_center_net', epoch)))
        self.net.polar_mask_net.load_state_dict(
            torch.load(root_path + '/%s/%s/%s_%.3d.pth' % ('experiment', exp, 'polar_mask_net', epoch)))
        self.net.backbone.eval()
        self.net.sem_net.eval()
        self.net.center_net.eval()
        self.net.polar_mask_net.eval()

    def eval(self, eval_dir, mass_analyzer, mz_resolution, resolution_mz, rt_fwhm, center_threshold=0.5, block_rt_width=None, block_mz_width=None, target_id=None):
        start_time = time.time()
        print('Evaluating on ', eval_dir)
        eval_file_paths = glob.glob(os.path.join(eval_dir, '*.csv'))
        print('Point cloud count: {}'.format(len(eval_file_paths)))
        result_list = []

        for file_dir in eval_file_paths:
            file_name = file_dir.split('/')[-1].split('.csv')[0]
            pc_id = int(file_name.split('_')[0])
            block_mz_center = float(file_name.split('_')[1])
            block_rt_center = float(file_name.split('_')[2])

            if target_id is not None and target_id != -1 and pc_id != target_id:
                continue

            reader = csv.reader(open(file_dir, 'r'))
            data = np.array(list(reader), dtype=np.float32)
            raw_points = data[:, :3]

            points = self.normalize_points(raw_points, mass_analyzer, mz_resolution, resolution_mz, rt_fwhm)
            # points = fill_feature_cuda(points)

            pc = torch.tensor(points.reshape(1, len(points), -1), dtype=torch.float32).cuda()

            """ feature extraction """
            point_features = self.net.backbone(pc[:, :, :3])
            """ semantic segmentation """
            pre_sem = self.net.sem_net(point_features)
            """ center prediction """
            pre_center = self.net.center_net(point_features).squeeze(-1)
            """ mask prediction """
            pre_masks = self.net.polar_mask_net(point_features)

            pre_sem = pre_sem[0].cpu().detach().numpy()
            pre_center = pre_center[0].cpu().detach().numpy()
            pre_masks = pre_masks[0].cpu().detach().numpy()

            center_idx = ((pre_center * pre_sem) > center_threshold).nonzero()[0]

            candidate_masks = pre_masks[center_idx]
            final_center_idx, final_masks = get_final_masks(points, pre_masks, center_idx, candidate_masks)

            if len(final_center_idx) == 0:
                continue
            arg_idx = np.argsort(-points[final_center_idx][:, 2])
            final_center_idx = final_center_idx[arg_idx]
            final_masks = np.array(final_masks)[arg_idx]

            # manage result
            point_instance = get_point_instance(points, final_center_idx, final_masks)
            for i in range(len(final_center_idx)):
                instance_idxes = (point_instance == i).nonzero()[0]
                if len(instance_idxes) < 10:
                    continue
                instance_points = raw_points[instance_idxes]

                # Intensity calculation
                total_intensity = self.get_instance_volume(instance_points)
                # total_intensity = np.sum(instance_points[:, 2])

                # Center point selection
                # apex_idx = instance_idxes[np.argmax(instance_points[:, 2])]
                # apex_raw_point = raw_points[apex_idx]
                apex_raw_point = raw_points[final_center_idx[i]]
                if block_rt_width is not None and block_mz_width is not None:
                    if apex_raw_point[0] >= block_rt_center + block_rt_width / 2 \
                            or apex_raw_point[0] < block_rt_center - block_rt_width / 2 \
                            or apex_raw_point[1] >= block_mz_center + block_mz_width / 2 \
                            or apex_raw_point[1] < block_mz_center - block_mz_width / 2:
                        continue

                result_list += [[pc_id, apex_raw_point[1], np.min(instance_points[:, 1]), np.max(instance_points[:, 1]),
                                apex_raw_point[0], np.min(instance_points[:, 0]), np.max(instance_points[:, 0]),
                                apex_raw_point[2], total_intensity, len(instance_points)]]

                # id, mz, mz_start, mz_end, rt, rt_start, rt_end, apex_intensity, volume, point_count, mask_len_0, ..., mask_len_35
                # print(total_intensity, apex_raw_point[0], apex_raw_point[1])

            print(file_dir, len(final_center_idx))

            # debug mode
            if target_id is not None:
                if target_id == -1 or pc_id == target_id:
                    from utils.visualize import Plot
                    print(len(center_idx))
                    Plot.draw_pc_heatmap(pc_xyz=points, idx=pc_id, heatmap=pre_center)
                    Plot.draw_pc_heatmap(pc_xyz=points, idx=pc_id, heatmap=pre_sem)
                    center_map = np.zeros(pre_center.shape)
                    center_map[center_idx] = 1
                    Plot.draw_pc_heatmap(pc_xyz=points, idx=pc_id, heatmap=center_map)
                    # Plot.draw_pc_polar(pc_xyzrgb=points, idx=pc_id, center_idxes=center_idx, polar_masks=candidate_masks)
                    Plot.draw_pc_polar(pc_xyzrgb=points, idx=pc_id, center_idxes=final_center_idx, polar_masks=final_masks)
                    print(pc_id)
        time_cost = time.time() - start_time
        print('Time Cost:', time_cost)

        if target_id is None:
            # output result
            output_file_name = eval_dir.split('/')[-1] + '-result-{}-{}.csv'.format(
                time.strftime("%Y%m%d_%H%M%S", time.localtime()), int(time_cost))
            output_file_dir = os.path.join(os.path.dirname(eval_dir), 'result')
            output_file_path = os.path.join(output_file_dir, output_file_name)
            if not os.path.exists(output_file_dir):
                os.mkdir(output_file_dir)

            output_file = open(output_file_path, 'w')
            writer = csv.writer(output_file)
            writer.writerows(result_list)
            output_file.close()
            print('Finish on ', output_file_path)

    def normalize_points(self, raw_points, mass_analyzer, mz_resolution, resolution_mz, rt_fwhm):
        points = raw_points.copy()
        min_rt = np.min(points[:, 0])
        max_rt = np.max(points[:, 0])
        min_mz = np.min(points[:, 1])
        max_mz = np.max(points[:, 1])
        mid_mz = (min_mz + max_mz) / 2
        mid_rt = (min_rt + max_rt) / 2

        mz_fwhm = get_mz_fwhm(mid_mz, mass_analyzer, mz_resolution, resolution_mz)
        rt_factor = 1.0 / rt_fwhm
        mz_factor = 1.0 / mz_fwhm
        points[:, 0] = (points[:, 0] - mid_rt) * rt_factor
        points[:, 1] = (points[:, 1] - mid_mz) * mz_factor
        points[:, 2] = np.log2(points[:, 2])
        min_intensity = np.min(points[:, 2]) - 1
        points[:, 2] -= min_intensity
        return points

    def get_instance_volume(self, instance_points):
        rt_sorted_points = instance_points[np.argsort(instance_points[:, 0])]
        rt_map = {}
        for point in rt_sorted_points:
            if not rt_map.__contains__(point[0]):
                rt_map[point[0]] = []
            rt_map[point[0]] += [point]
        rt_list = list(rt_map.keys())
        if len(rt_list) < 2:
            return 0
        areas = []
        for i in range(len(rt_list) - 1, -1, -1):
            rt = rt_list[i]
            tmp_points = np.array(rt_map[rt])
            if len(tmp_points) < 2:
                rt_list.remove(rt)
                continue
            mz_sorted_points = tmp_points[np.argsort(tmp_points[:, 1])]
            mz_interval = mz_sorted_points[1:, 1] - mz_sorted_points[:-1, 1]
            mid_intensity = (mz_sorted_points[1:, 2] + mz_sorted_points[:-1, 2]) / 2
            area = np.sum(mz_interval * mid_intensity)
            areas += [area]
        rt_list = np.array(rt_list)
        areas = np.array(areas)
        rt_interval = rt_list[1:] - rt_list[:-1]
        mid_area = (areas[1:] + areas[:-1]) / 2
        volume = np.sum(rt_interval * mid_area)
        return volume

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Untargeted feature extraction')

    parser.add_argument('--data_dir', type=str, help='dataset dir', required=True)
    parser.add_argument('--mass_analyzer', type=str, help='orbitrap or tof', required=True)
    parser.add_argument('--mz_resolution', type=float, help='the resolution of mass analyzer', required=True)
    parser.add_argument('--resolution_mz', type=float, help='the m/z value at the resolution', required=True)
    parser.add_argument('--rt_fwhm', type=float, help='median of feature RT FWHM', required=True)
    parser.add_argument('--experiment', type=str, help='choose a pretrained model', default='msnet_20220215_143158')
    parser.add_argument('--epoch', type=int, help='choose the epoch of saved model', default=300)
    parser.add_argument('--center_threshold', type=float, help='feature center selection threshold', default=0.5)
    parser.add_argument('--block_rt_width', type=float, help='point cloud rt window width', default=6)
    parser.add_argument('--block_mz_width', type=float, help='point cloud m/z window width', default=0.8)
    parser.add_argument('--target_id', type=int, help='None, not visualize. -1, visualize each point cloud. Integer, visualize a specific point cloud', default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    data_dir = glob.glob(os.path.join(args.data_dir, '*arget-*'))
    print(data_dir)
    evaluator = MsNetEvaluator(exp=args.experiment, epoch=args.epoch)
    for eval_dir in data_dir:
        evaluator.eval(eval_dir=eval_dir, mass_analyzer=args.mass_analyzer, mz_resolution=args.mz_resolution,
                       resolution_mz=args.resolution_mz, rt_fwhm=args.rt_fwhm, center_threshold=args.center_threshold,
                       block_rt_width=args.block_rt_width, block_mz_width=args.block_mz_width, target_id=args.target_id)
