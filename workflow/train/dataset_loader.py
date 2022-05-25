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
import csv
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from torch.utils.data import DataLoader
from utils.polar_mask import get_polar_mask
from utils.ms_compatibility import get_mz_fwhm
import model.msnet_utils as msnet_utils


class Dataset:
    def __init__(self, cfg):

        self.data_root = os.path.join(ROOT_DIR, cfg.data_root)
        self.dataset = cfg.dataset
        self.data_sim_dir = cfg.data_sim_dir
        self.data_anno_dir = cfg.data_anno_dir

        self.batch_size = cfg.batch_size

        self.train_workers = cfg.train_workers
        self.val_workers = cfg.val_workers

        self.train_list_suffix = cfg.train_list_suffix
        self.val_list_suffix = cfg.val_list_suffix
        self.test_list_suffix = cfg.test_list_suffix

        self.max_nins = cfg.max_nins

        self.train_file_data = []
        self.val_file_data = []


    def train_sim_loader(self):
        train_file_names = open(os.path.join(self.data_root, self.dataset, self.data_sim_dir + self.train_list_suffix),
                                'r').readlines()
        self.train_files = [os.path.join(self.data_root, self.dataset, self.data_sim_dir, i.strip()) for i in
                            train_file_names]

        for file_dir in self.train_files:
            reader = csv.reader(open(file_dir, 'r'))
            data = np.array(list(reader), dtype=np.float32)
            points = data[:, :3]
            ins_labels = data[:, 3]

            points = self.normalize_points(points)
            points = fill_feature_cuda(points)

            self.train_file_data += [(points, ins_labels)]

        train_set = list(range(len(self.train_file_data)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.train_merge,
                                            num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=False)

    def train_anno_loader(self):
        train_file_names = open(os.path.join(self.data_root, self.dataset, self.data_anno_dir + self.train_list_suffix),
                                'r').readlines()
        self.train_files = [os.path.join(self.data_root, self.dataset, self.data_anno_dir, i.strip()) for i in
                            train_file_names]

        for file_dir in self.train_files:
            reader = csv.reader(open(file_dir, 'r'))
            data = np.array(list(reader), dtype=np.float32)
            points = data[:, :3]
            ins_labels = data[:, 3]

            points = self.normalize_anno_points(points)
            points = fill_feature_cuda(points)

            dense_idx = points[:, -1] > 0.3
            points = points[dense_idx]
            ins_labels = ins_labels[dense_idx]

            self.train_file_data += [(points, ins_labels)]

        train_set = list(range(len(self.train_file_data)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.train_merge,
                                            num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=False)

    def val_sim_loader(self):
        val_file_names = open(os.path.join(self.data_root, self.dataset, self.data_sim_dir + self.val_list_suffix),
                              'r').readlines()
        self.val_files = [os.path.join(self.data_root, self.dataset, self.data_sim_dir, i.strip()) for i in val_file_names]

        for file_dir in self.val_files:
            reader = csv.reader(open(file_dir, 'r'))
            data = np.array(list(reader), dtype=np.float32)
            points = data[:, :3]
            ins_labels = data[:, 3]
            points = self.normalize_points(points)

            points = fill_feature_cuda(points)

            self.val_file_data += [(points, ins_labels)]
        val_set = list(range(len(self.val_file_data)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.val_merge,
                                          num_workers=self.val_workers,
                                          shuffle=False, drop_last=True, pin_memory=False)

    def val_anno_loader(self):
        val_file_names = open(os.path.join(self.data_root, self.dataset, self.data_anno_dir + self.val_list_suffix),
                              'r').readlines()
        self.val_files = [os.path.join(self.data_root, self.dataset, self.data_anno_dir, i.strip()) for i in val_file_names]

        for file_dir in self.val_files:
            reader = csv.reader(open(file_dir, 'r'))
            data = np.array(list(reader), dtype=np.float32)
            points = data[:, :3]
            ins_labels = data[:, 3]
            points = self.normalize_anno_points(points)

            points = fill_feature_cuda(points)

            self.val_file_data += [(points, ins_labels)]
        val_set = list(range(len(self.val_file_data)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.val_merge,
                                          num_workers=self.val_workers,
                                          shuffle=False, drop_last=True, pin_memory=False)

    def testLoader(self):
        print("todo")

    def train_merge(self, id):
        batch_points = []
        batch_labels = []
        batch_center_idxes = []
        batch_polar_masks = []
        batch_center_heatmaps = []
        for i, idx in enumerate(id):
            points, ins_labels = self.train_file_data[idx]

            # scale
            scale = (np.random.random(3) * 1 + 1) ** (np.random.binomial(1, 0.5) * 2 - 1)
            scale = np.concatenate((scale, np.ones(points.shape[-1] - 3)))
            scale = scale.reshape(1, -1)
            points = points * scale
            # offset
            points[:, 2] += np.random.random(len(points))
            offset = (np.random.random(2).reshape(1, 2) - 0.5) * 10
            points[:, :2] += offset

            batch_points += [points]
            batch_labels += [ins_labels]

        ### merge all the scenes in the batchd
        max_point_num = max(len(points) for points in batch_points)
        for i in range(len(batch_points)):
            points = batch_points[i]
            ins_labels = batch_labels[i]
            fill_len = max_point_num - len(points)
            if fill_len != 0:
                batch_points[i], batch_labels[i] = get_noise_to_fill(points, ins_labels, fill_len)

            sort_idxes = np.argsort(-batch_points[i][:, 2])
            batch_points[i] = batch_points[i][sort_idxes, :]
            batch_labels[i] = batch_labels[i][sort_idxes]

            center_idxes, polar_masks, center_heatmap = get_polar_mask(batch_points[i], batch_labels[i])
            if len(center_idxes) != 0:
                batch_center_idxes += [
                    np.concatenate((center_idxes, np.ones(self.max_nins - len(center_idxes)) * -1), axis=0)]
                batch_polar_masks += [
                    np.concatenate((polar_masks, np.zeros((self.max_nins - len(center_idxes), polar_masks.shape[1]))),
                                   axis=0)]
            else:
                batch_center_idxes += [np.ones(self.max_nins) * -1]
                batch_polar_masks += [np.zeros((self.max_nins, 36))]
            batch_center_heatmaps += [center_heatmap]


        batch_points = np.array(batch_points).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int)
        batch_center_idxes = np.array(batch_center_idxes).astype(np.int)
        batch_polar_masks = np.array(batch_polar_masks).astype(np.float32)
        batch_center_heatmaps = np.array(batch_center_heatmaps).astype(np.float32)
        return torch.tensor(batch_points, dtype=torch.float32), torch.tensor(batch_labels, dtype=torch.long), \
               torch.tensor(batch_center_idxes, dtype=torch.long), torch.tensor(batch_polar_masks, dtype=torch.float32),\
               torch.tensor(batch_center_heatmaps, dtype=torch.float32)

    def val_merge(self, id):
        batch_points = []
        batch_labels = []
        batch_center_idxes = []
        batch_polar_masks = []
        batch_center_heatmaps = []
        for i, idx in enumerate(id):
            points, ins_labels = self.val_file_data[idx]
            batch_points += [points]
            batch_labels += [ins_labels]

        ### merge all the scenes in the batchd
        min_point_num = min(len(points) for points in batch_points)
        max_point_num = max(len(points) for points in batch_points)
        for i in range(len(batch_points)):
            points = batch_points[i]
            ins_labels = batch_labels[i]

            fill_len = max_point_num - len(points)

            if fill_len != 0:
                batch_points[i], batch_labels[i] = get_noise_to_fill(points, ins_labels, fill_len)

            sort_idxes = np.argsort(-batch_points[i][:, 2])
            batch_points[i] = batch_points[i][sort_idxes, :]
            batch_labels[i] = batch_labels[i][sort_idxes]

            center_idxes, polar_masks, center_heatmap = get_polar_mask(batch_points[i], batch_labels[i])
            if len(center_idxes) != 0:
                batch_center_idxes += [
                    np.concatenate((center_idxes, np.ones(self.max_nins - len(center_idxes)) * -1), axis=0)]
                batch_polar_masks += [
                    np.concatenate((polar_masks, np.zeros((self.max_nins - len(center_idxes), polar_masks.shape[1]))),
                                   axis=0)]
            else:
                batch_center_idxes += [np.ones(self.max_nins) * -1]
                batch_polar_masks += [np.zeros((self.max_nins, 36))]
            batch_center_heatmaps += [center_heatmap]

        batch_points = np.array(batch_points).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.int)
        batch_center_idxes = np.array(batch_center_idxes).astype(np.int)
        batch_polar_masks = np.array(batch_polar_masks).astype(np.float32)
        batch_center_heatmaps = np.array(batch_center_heatmaps).astype(np.float32)
        return torch.tensor(batch_points, dtype=torch.float32), torch.tensor(batch_labels, dtype=torch.long), \
               torch.tensor(batch_center_idxes, dtype=torch.long), torch.tensor(batch_polar_masks, dtype=torch.float32), \
               torch.tensor(batch_center_heatmaps, dtype=torch.float32)

    def normalize_anno_points(self, points):
        min_rt = np.min(points[:, 0])
        max_rt = np.max(points[:, 0])
        min_mz = np.min(points[:, 1])
        max_mz = np.max(points[:, 1])
        min_intensity = np.min(points[:, 2])
        max_intensity = np.max(points[:, 2])

        mz_factor = get_mz_fwhm((min_mz + max_mz) / 200, 'tof', 35000, 956) * 100
        rt_factor = 0.1 * 60
        points[:, 0] -= (min_rt + max_rt) / 2
        points[:, 0] /= rt_factor
        points[:, 1] -= (min_mz + max_mz) / 2
        points[:, 1] /= mz_factor
        min_intensity = np.min(points[:, 2]) - 1
        points[:, 2] -= min_intensity
        return points

    def normalize_points(self, points):
        min_intensity = np.min(points[:, 2])
        points[:, 2] -= min_intensity
        return points


def get_noise_to_fill(points, labels, fill_length):
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])
    noise_points = np.random.rand(fill_length, 3)
    noise_points[:, 2] = np.min([5 * noise_points[:, 2], (0.5 - np.abs(noise_points[:, 1] - 0.5)) * 10], axis=0) + 1
    noise_points[:, 0] = noise_points[:, 0] * (max_x - min_x) + min_x
    noise_points[:, 1] = noise_points[:, 1] + max_y
    noise_points = np.concatenate((noise_points, np.zeros((fill_length, points.shape[-1] - 3))), axis=-1)

    points = np.concatenate((points, noise_points), axis=0)
    labels = np.concatenate((labels, np.ones(fill_length) * -1), axis=0)
    return points, labels


# del noise for collate_fn
# assert noise_len is larger than del_length
def get_del_idx(points, labels, del_length):
    low_indexes = (points[:, 2] < 8).nonzero()[0]
    return low_indexes[np.random.choice(np.arange(0, len(low_indexes)), del_length, replace=False)]
    # return (labels == -1).nonzero()[0][:del_length]


# fill density feature
def fill_feature(points, x_tolerance=0.5, y_tolerance=0.5):
    min_x = points[:, 0] - x_tolerance
    max_x = points[:, 0] + x_tolerance
    min_y = points[:, 1] - y_tolerance
    max_y = points[:, 1] + y_tolerance
    point_matrix = np.expand_dims(points, 1).repeat(len(points), axis=1)
    neighbor_matrix = (point_matrix[:, :, 0] > min_x) * (point_matrix[:, :, 0] < max_x) *\
                      (point_matrix[:, :, 1] > min_y) * (point_matrix[:, :, 1] < max_y)
    cnt = np.sum(neighbor_matrix, axis=-1)
    max_cnt = np.max(cnt).astype(np.float32)
    density = cnt / max_cnt
    density = density.reshape(-1, 1)
    points = np.concatenate((points, density), axis=-1)
    return points


def fill_feature_cuda(points, radius=None):
    if radius is None:
        radius = [0.2, 0.3, 0.4]
    xyz = torch.tensor(points, dtype=torch.float32).cuda().unsqueeze(0)
    total_density = []
    for i in range(len(radius)):
        idx = msnet_utils.ms_query(radius[i], 100, xyz, xyz).squeeze(0).cpu().numpy()
        cnt = np.sum(idx - idx[:, 0:1] != 0, axis=1) + 1.0
        max_cnt = np.max(cnt)
        density = cnt / max_cnt
        total_density += [density]
    max_density = np.max(np.array(total_density), axis=0)
    points = np.concatenate((points, max_density.reshape(-1, 1)), axis=-1)
    return points
