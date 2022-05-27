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
import glob
import json
import csv
import math
from random import shuffle

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from utils.config import cfg
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


class SimulateDataGenerator:
    def __init__(self):
        # base
        self.base_rt_left = -10
        self.base_rt_right = 10
        self.base_mz_left = -10
        self.base_mz_right = 10
        self.min_rt_step = 0.03
        self.max_rt_step = 0.06
        self.min_mz_step = 0.02
        self.max_mz_step = 0.08

        # peak
        self.min_peak_height = 2 ** 7
        self.mid_peak_height = 2 ** 12
        self.max_peak_height = 2 ** 20
        self.min_peak_rt_left_width = 0.2
        self.mid_peak_rt_left_width = 0.5
        self.max_peak_rt_left_width = 1
        self.min_peak_rt_right_width_factor = 0.7
        self.mid_peak_rt_right_width_factor = 1.5
        self.max_peak_rt_right_width_factor = 4
        self.min_peak_mz_left_width = 0.4
        self.mid_peak_mz_left_width = 0.5
        self.max_peak_mz_left_width = 0.6
        self.min_peak_mz_right_width_factor = 1
        self.mid_peak_mz_right_width_factor = 1.2
        self.max_peak_mz_right_width_factor = 1.5
        self.max_peaks = 25
        self.peaks = []

        # noise
        self.min_noise_factor = 0
        self.mid_noise_factor = 0
        self.max_noise_factor = 8

        # points
        self.min_label_height = 2 ** 6
        self.filter_noise = 2 ** 6

    # Main Function
    def generate(self):
        self.prepare_base()
        self.fill_noise()
        self.fill_peaks()
        self.screen_to_points()

    def prepare_base(self):
        self.rt_step = self.rand_random_distribution(self.min_rt_step, self.max_rt_step)
        self.rt_values = np.arange(self.base_rt_left, self.base_rt_right, self.rt_step)

        self.mz_step = self.rand_random_distribution(self.min_mz_step, self.max_mz_step)
        self.mz_values = np.arange(self.base_mz_left, self.base_mz_right, self.mz_step)
        self.base = np.zeros((len(self.rt_values), len(self.mz_values)))
        self.base_label = np.ones(self.base.shape) * -1

    def fill_noise(self):
        for i in range(len(self.base)):
            for j in range(len(self.base[i])):
                self.base[i, j] = 2 ** self.rand_normal_distribution(self.min_noise_factor, self.mid_noise_factor,
                                                                self.max_noise_factor)

    def fill_peaks(self):
        # random peak positions
        positions = np.random.rand(self.max_peaks, 2)
        positions[:, 0] = positions[:, 0] * (self.base_rt_right - self.base_rt_left) + self.base_rt_left
        positions[:, 1] = positions[:, 1] * (self.base_mz_right - self.base_mz_left) + self.base_mz_left

        peak_label = 0
        for left_bottom_position in positions:
            rt_left_width, rt_right_width, mz_left_width, mz_right_width, peak_height = self.get_peak_params()
            mz_len = mz_left_width + mz_right_width
            rt_len = rt_left_width + rt_right_width
            peak_direct = np.array([rt_len, mz_len])
            right_top_position = left_bottom_position + peak_direct

            # ignore peak out of bounds
            if right_top_position[0] > self.base_rt_right or right_top_position[1] > self.base_mz_right:
                continue

            # ignore cross positions
            invalid_position = False
            for peak in self.peaks:
                root = peak[0]
                direct = peak[1]
                if root[0] > left_bottom_position[0]:
                    right_root = root
                    right_direct = direct
                    left_root = left_bottom_position
                    left_direct = peak_direct
                else:
                    right_root = left_bottom_position
                    right_direct = peak_direct
                    left_root = root
                    left_direct = direct
                if left_root[1] > right_root[1]:
                    if left_root[1] < right_root[1] + right_direct[1] and left_root[0] + left_direct[0] > right_root[0]:
                        invalid_position = True
                else:
                    if right_root[0] < left_root[0] + left_direct[0] and right_root[1] < left_direct[1] + left_root[1]:
                        invalid_position = True
            if invalid_position:
                continue

            # insert peak
            rt_left_index = math.ceil((left_bottom_position[0] - self.base_rt_left) / self.rt_step)
            rt_right_index = math.floor((right_top_position[0] - self.base_rt_left) / self.rt_step)
            mz_bottom_index = math.ceil((left_bottom_position[1] - self.base_mz_left) / self.mz_step)
            mz_up_index = math.floor((right_top_position[1] - self.base_mz_left) / self.mz_step)
            rt_heights = self.sample_from_gaussian(rt_left_width, rt_right_width, left_bottom_position[0] + rt_left_width,
                                              peak_height, self.rt_values, np.arange(rt_left_index, rt_right_index + 1))
            for i in range(len(rt_heights)):
                rt_index = rt_left_index + i
                rt_height = rt_heights[i]
                mz_indexes = np.arange(mz_bottom_index, mz_up_index + 1)
                mz_heights = self.sample_from_gaussian(mz_left_width, mz_right_width,
                                                  left_bottom_position[1] + mz_left_width,
                                                  rt_height, self.mz_values, mz_indexes)
                self.base[rt_index, mz_indexes] = self.base[rt_index, mz_indexes] + mz_heights
                label_mz_indexes = mz_indexes[mz_heights >= self.min_label_height]
                self.base_label[rt_index, label_mz_indexes] = peak_label

            self.peaks += [[left_bottom_position, peak_direct]]
            peak_label += 1

    def screen_to_points(self):
        # valid_status = (self.base_label >= 0) + (self.base > self.filter_noise)
        valid_status = self.base > self.filter_noise
        valid_coords = valid_status.nonzero()
        rt_indexes = valid_coords[0]
        mz_indexes = valid_coords[1]
        self.result_points = np.vstack((self.rt_values[rt_indexes], self.mz_values[mz_indexes],
                                        np.log2(self.base[valid_status]))).transpose()
        self.result_labels = self.base_label[valid_status]

    def get_peak_params(self):
        height = self.rand_normal_distribution(self.min_peak_height, self.mid_peak_height, self.max_peak_height)

        rt_left_width = self.rand_normal_distribution(self.min_peak_rt_left_width, self.mid_peak_rt_left_width
                                                 , self.max_peak_rt_left_width)
        rt_right_width = self.rand_normal_distribution(self.min_peak_rt_right_width_factor,
                                                  self.mid_peak_rt_right_width_factor,
                                                  self.max_peak_rt_right_width_factor) * rt_left_width

        mz_left_width = self.rand_normal_distribution(self.min_peak_mz_left_width, self.mid_peak_mz_left_width,
                                                 self.max_peak_mz_left_width)
        mz_right_width = self.rand_normal_distribution(self.min_peak_mz_right_width_factor,
                                                  self.mid_peak_mz_right_width_factor,
                                                  self.max_peak_mz_right_width_factor) * mz_left_width

        return rt_left_width, rt_right_width, mz_left_width, mz_right_width, height

    def rand_random_distribution(self, from_value, to_value):
        random = np.random.random()
        return from_value + (to_value - from_value) * random

    def rand_normal_distribution(self, from_value, top_value, to_value):
        random = np.random.randn()
        while random < -3 or random > 3:
            random = np.random.randn()
        if random < 0:
            return top_value + (top_value - from_value) * random / 3
        else:
            return top_value + (to_value - top_value) * random / 3

    def sample_from_gaussian(self, left_width, right_width, mid, height, values, indexes):
        left_sigma = left_width / 4
        right_sigma = right_width / 4
        left_power_coef = -1 / (2 * left_sigma ** 2)
        right_power_coef = -1 / (2 * right_sigma ** 2)

        sample_result = np.zeros(len(indexes))
        for i in range(len(indexes)):
            value = values[indexes[i]]
            if value < mid:
                sample_result[i] = height * np.exp((value - mid) ** 2 * left_power_coef)
            else:
                sample_result[i] = height * np.exp((value - mid) ** 2 * right_power_coef)

        return sample_result


def generate_sim_dataset():
    output_dir = os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, cfg.data_sim_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(cfg.data_sim_num):
        output_file_path = os.path.join(BASE_DIR, output_dir, str(i + 1) + "_sim.csv")
        print(output_file_path)
        output_file = open(output_file_path, 'w')

        sim_generator = SimulateDataGenerator()
        sim_generator.generate()
        points = sim_generator.result_points
        labels = sim_generator.result_labels
        point_with_anno = [np.concatenate((points[i], [labels[i]])) for i in range(len(points))]
        writer = csv.writer(output_file)
        writer.writerows(point_with_anno)

        output_file.close()


def generate_anno_dataset():
    anno_paths = glob.glob(os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, cfg.anno_dir, '*.json'))
    output_dir = os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, cfg.data_anno_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for anno_path in anno_paths:
        file_name = os.path.split(anno_path)[1].split('.json')[0]
        raw_path = os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, cfg.raw_dir, file_name + '.pcd')
        output_file_path = os.path.join(BASE_DIR, output_dir, file_name + '.csv')
        print(output_file_path)
        output_file = open(output_file_path, 'w')

        point_cloud = load_raw_file(raw_path)
        anno = load_anno_file(anno_path, len(point_cloud))
        point_with_anno = [(point_cloud[i] + [anno[i]]) for i in range(len(point_cloud))]
        writer = csv.writer(output_file)
        writer.writerows(point_with_anno)

        output_file.close()


def split_dataset(data_dir):
    data_files = glob.glob(os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, data_dir, '*.csv'))
    names = [os.path.split(file)[1] for file in data_files]
    shuffle(names)
    train_num = math.floor(len(names) * cfg.train_percent)
    test_num = math.floor(len(names) * cfg.test_percent)
    val_num = len(names) - train_num - test_num

    train_file_dir = os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, data_dir + cfg.train_list_suffix)
    train_file = open(train_file_dir, 'w')
    train_file.writelines([line + '\n' for line in names[: train_num]])
    train_file.close()

    val_file_dir = os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, data_dir + cfg.val_list_suffix)
    val_file = open(val_file_dir, 'w')
    val_file.writelines([line + '\n' for line in names[train_num: (train_num + val_num)]])
    val_file.close()

    test_file_dir = os.path.join(BASE_DIR, cfg.data_root, cfg.dataset, data_dir + cfg.test_list_suffix)
    test_file = open(test_file_dir, 'w')
    test_file.writelines([line + '\n' for line in names[(train_num + val_num): (train_num + val_num + test_num)]])
    test_file.close()


def load_raw_file(file_path):
    file = open(file_path, 'r')
    point_cloud = []
    for line in file.readlines():
        line = line.split()
        if line[0].isalpha():
            continue
        point_cloud += [line]
    return point_cloud


def load_anno_file(file_path, point_num):
    file = open(file_path, 'r')
    file = json.load(file)
    peak_count = len(file['result']['data'])
    anno = [-1] * point_num
    for i in range(peak_count):
        peak = file['result']['data'][i]['indexs']
        for index in peak:
            anno[index] = i
    return anno


if __name__ == '__main__':
    # generate_anno_dataset()

    split_dataset(cfg.data_anno_dir)
