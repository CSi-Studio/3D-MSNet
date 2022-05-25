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
import os
from open3d import linux as open3d  ## pip install open3d-python==0.3.0
import random
import colorsys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from utils.polar_mask import get_point_instance


class Plot:

    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb, idx, bboxes = np.array([])):
        pc = open3d.PointCloud()
        # top_pc = pc_xyzrgb.copy()
        # top_pc[:, 2] = np.log(top_pc[:, 2])
        # pc.points = open3d.Vector3dVector(top_pc[:, 0:3])
        pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            intensity = pc_xyzrgb[:, 2]
            max_intensity = 10
            min_intensity = -2
            colors = np.zeros((pc_xyzrgb.shape[0], 3))
            for i, value in enumerate(pc_xyzrgb[:, 2]):
                grey = min(1.0, max(0.0, value - min_intensity) / (max_intensity - min_intensity))
                colors[i, 0] = 1 - grey
                colors[i, 1] = 1 - grey
                colors[i, 2] = 1 - grey
            pc.colors = open3d.Vector3dVector(colors)
        elif np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.Vector3dVector(pc_xyzrgb[:, 3:6])
        series = [pc]
        if bboxes.any():
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                     [4, 5], [5, 6], [6, 7], [4, 7],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            for i, bbox in enumerate(bboxes):
                corner_points = [[bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2, bbox[2] - bbox[5] / 2],
                                 [bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2, bbox[2] + bbox[5] / 2],
                                 [bbox[0] - bbox[3] / 2, bbox[1] + bbox[4] / 2, bbox[2] + bbox[5] / 2],
                                 [bbox[0] - bbox[3] / 2, bbox[1] + bbox[4] / 2, bbox[2] - bbox[5] / 2],
                                 [bbox[0] + bbox[3] / 2, bbox[1] - bbox[4] / 2, bbox[2] - bbox[5] / 2],
                                 [bbox[0] + bbox[3] / 2, bbox[1] - bbox[4] / 2, bbox[2] + bbox[5] / 2],
                                 [bbox[0] + bbox[3] / 2, bbox[1] + bbox[4] / 2, bbox[2] + bbox[5] / 2],
                                 [bbox[0] + bbox[3] / 2, bbox[1] + bbox[4] / 2, bbox[2] - bbox[5] / 2]]
                # if i == len(bboxes) // 2:
                #     colors = [[1, 0, 0] for _ in range(len(lines))]
                # else:
                colors = [[0, 0, 1] for _ in range(len(lines))]
                line_set = open3d.LineSet()
                line_set.points = open3d.Vector3dVector(corner_points)
                line_set.lines = open3d.Vector2iVector(lines)
                line_set.colors = open3d.Vector3dVector(colors)
                series += [line_set]
        # series += [open3d.create_mesh_coordinate_frame(size=15, origin=[-20, -20, 0])]
        open3d.draw_geometries(series, window_name=str(idx))
        return 0

    @staticmethod
    def draw_pc_center(pc_xyzrgb, idx, center_xys, scores):
        pc = open3d.PointCloud()
        pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])

        series = [pc]
        lines = [[0, 1]]
        for i in range(center_xys.shape[0]):
            xy = center_xys[i]
            score = scores[i]
            support_points = np.array([[xy[0], xy[1], 0], [xy[0], xy[1], 20]])
            colors = [[score, 0, 0]]
            line_set = open3d.LineSet()
            line_set.points = open3d.Vector3dVector(support_points)
            line_set.lines = open3d.Vector2iVector(lines)
            line_set.colors = open3d.Vector3dVector(colors)
            series += [line_set]

        open3d.draw_geometries(series, window_name=str(idx))
        return 0

    @staticmethod
    def draw_pc_polar(pc_xyzrgb, idx, center_idxes, polar_masks=np.array([])):
        angle_num = polar_masks.shape[-1]
        pc = open3d.PointCloud()
        # top_pc = pc_xyzrgb.copy()
        # top_pc[:, 2] = 0
        # pc.points = open3d.Vector3dVector(top_pc[:, 0:3])
        pc.points = open3d.Vector3dVector(pc_xyzrgb[:, 0:3])
        ins_colors = Plot.random_colors(len(polar_masks), seed=2)
        point_instance = get_point_instance(pc_xyzrgb, center_idxes, polar_masks)

        pc_colors = np.zeros((point_instance.shape[0], 3))
        for i in range(len(center_idxes)):
            pc_colors[point_instance == i] = ins_colors[i]
        pc.colors = open3d.Vector3dVector(pc_colors)
        series = [pc]
        if polar_masks.any():
            center_line = [[0,1]]
            lines = [[i, i + 1] for i in range(angle_num - 1)]
            lines += [[angle_num - 1, 0]]
            angles = np.arange(-np.pi, np.pi, np.pi * 2. / angle_num)


            for i, mask in enumerate(polar_masks):
                center = pc_xyzrgb[center_idxes[i], :2]
                support_points = np.repeat([center], angle_num, axis=0) + np.array([mask * np.cos(angles), mask * np.sin(angles)]).transpose()
                support_points = np.concatenate((support_points, np.ones((support_points.shape[0], 1)) * -0.5), axis=1)
                colors = [ins_colors[i] for _ in range(len(lines))]
                mask_set = open3d.LineSet()
                mask_set.points = open3d.Vector3dVector(support_points)
                mask_set.lines = open3d.Vector2iVector(lines)
                mask_set.colors = open3d.Vector3dVector(colors)
                # series += [mask_set]

                support_points = np.concatenate((np.array([[center[0], center[1], -0.5]]), support_points), axis=0)
                center_line = [[j, 0] for j in range(1, 37)]
                colors = [ins_colors[i] for _ in range(len(lines))]
                # support_points = np.array([[center[0], center[1], -0.5], [center[0], center[1], 20]])
                # colors = [ins_colors[i]]
                line_set = open3d.LineSet()
                line_set.points = open3d.Vector3dVector(support_points)
                line_set.lines = open3d.Vector2iVector(center_line)
                line_set.colors = open3d.Vector3dVector(colors)
                # series += [line_set]

        open3d.draw_geometries(series, window_name=str(idx))

        return 0

    @staticmethod
    def draw_pc_semins(pc_xyz, pc_semins, idx, fix_color_num=None, sem=0, bboxes=np.array([])):
        pc_xyz = pc_xyz[pc_xyz[:, 0].nonzero()]
        pc_semins = pc_semins[pc_xyz[:, 0].nonzero()]
        if fix_color_num is not None:
            ins_colors = Plot.random_colors(fix_color_num + 1, seed=2)
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_semins)) + 1, seed=2)  # cls 14

        semins_labels = np.unique(pc_semins)
        semins_bbox = []
        Y_colors = np.zeros((pc_semins.shape[0], 3))
        for id, semins in enumerate(semins_labels):

            valid_ind = np.argwhere(pc_semins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if fix_color_num is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            # bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0])
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1])
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2])
            zmax = np.max(valid_xyz[:, 2])
            semins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins, idx, bboxes)
        return Y_semins

    @staticmethod
    def draw_pc_heatmap(pc_xyz, heatmap, idx):
        heatmap = heatmap[pc_xyz[:, 0].nonzero()]
        pc_xyz = pc_xyz[pc_xyz[:, 0].nonzero()]
        arg_idx = np.argsort(heatmap)
        pc_xyz = pc_xyz[arg_idx]
        heatmap = heatmap[arg_idx]
        Y_colors = np.zeros((pc_xyz.shape[0], 3))
        max_intensity = 10
        min_intensity = -2
        for i, value in enumerate(pc_xyz[:, 2]):
            grey = min(1.0, max(0.0, value - min_intensity) / (max_intensity - min_intensity))
            Y_colors[i, 0] = (1 - grey) * 255
            Y_colors[i, 1] = (1 - grey) * 255
            Y_colors[i, 2] = (1 - grey) * 255
        for i, value in enumerate(heatmap):
            # if value <= 0.5:
            #     b = 255 * (1 - 2 * value)
            #     Y_colors[i] = np.array([255 - b, 255 - b, 255 - b])
            # else:
            #     r = 255 * (2 * value - 1)
            #     Y_colors[i] = np.array([255, 255 - r, 255 - r])
            factor_down = 0.5
            factor_up = 0.7
            if value < factor_down:
                continue
            value = (max(min(value, factor_up), factor_down) - factor_down) / (factor_up - factor_down)
            Y_colors[i][0] = max(255 * value, Y_colors[i][1])
        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins, idx)
        return Y_semins


def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    open3d.draw_geometries_with_animation_callback(pcd, rotate_view)
