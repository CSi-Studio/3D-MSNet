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
import torch


def get_polar_mask(points, ins_labels):
    points = points[:, :3]
    unique_labels = np.unique(ins_labels)
    center_idx = []
    polar_masks = []
    center_heatmap = np.zeros((len(points)))
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        instance_idxes = (ins_labels == label).nonzero()[0]
        if len(instance_idxes) < 10:
            continue
        instance_points = points[instance_idxes]
        ins_center_idx = np.argmax(instance_points[:, 2])
        center_point = instance_points[ins_center_idx]
        center_idx += [instance_idxes[ins_center_idx]]

        support_points = _get_support_points(instance_points)
        polar_mask = _support_to_mask(support_points, instance_points[ins_center_idx, :2], 36) * 1.1
        polar_masks += [polar_mask]

        dists = np.sqrt(np.sum((instance_points - center_point) ** 2, axis=-1))
        center_heatmap[instance_idxes] = _sample_from_gaussian(dists, heatmap=center_heatmap[instance_idxes], radius=0.2)

    return np.array(center_idx), np.array(polar_masks), center_heatmap


def points_in_mask(points, center, mask):
    ray_num = len(mask)
    unit_angle = np.pi * 2 / ray_num

    # point relative position to center
    relative_direct = (points[:, :2] - center[:2])
    relative_angle = np.arctan2(relative_direct[:, 1], relative_direct[:, 0])
    relative_len = np.sqrt(np.sum(relative_direct ** 2, axis=1))

    lower_angle = relative_angle - np.floor(relative_angle / unit_angle) * unit_angle
    higher_angle = unit_angle - lower_angle

    lower_idx = np.floor(relative_angle / unit_angle).astype(np.int) + int(ray_num / 2)
    lower_idx[lower_idx == -1] = ray_num - 1
    lower_idx[lower_idx == ray_num] = 0
    higher_idx = lower_idx + 1
    higher_idx[higher_idx == ray_num] = 0

    lower_len = mask[lower_idx]
    higher_len = mask[higher_idx]

    in_mask = relative_len <= ((lower_len * higher_angle + higher_len * lower_angle) / unit_angle)
    # in_mask = relative_len <= calc_angle_len(higher_angle, lower_angle, higher_len, lower_len)
    return in_mask


def points_in_mask_cuda(points, center, mask):
    ray_num = len(mask)
    unit_angle = np.pi * 2 / ray_num

    # point relative position to center
    relative_direct = (points[:, :2] - center[:2])
    relative_direct[relative_direct[:, 0] == 0, 0] = 1e-8
    relative_angle = torch.arctan(relative_direct[:, 1] / relative_direct[:, 0])
    relative_len = torch.sqrt(torch.sum(relative_direct ** 2, dim=1))

    lower_angle = relative_angle - torch.floor(relative_angle / unit_angle) * unit_angle
    higher_angle = unit_angle - lower_angle

    lower_idx = torch.floor(relative_angle / unit_angle).long() + int(ray_num / 2)
    lower_idx[lower_idx == -1] = ray_num - 1
    lower_idx[lower_idx == ray_num] = 0
    higher_idx = lower_idx + 1
    higher_idx[higher_idx == ray_num] = 0

    lower_len = mask[lower_idx]
    higher_len = mask[higher_idx]

    in_mask = relative_len <= ((lower_len * higher_angle + higher_len * lower_angle) / unit_angle)
    # in_mask = relative_len <= calc_angle_len(higher_angle, lower_angle, higher_len, lower_len)
    return in_mask


def get_final_masks(pc, pre_masks, center_idx, masks):
    # pre treatment
    candidate_center_idx = center_idx
    candidate_masks = masks
    candidate_masks[candidate_masks < 0.01] = 0.01
    if len(candidate_masks) == 0:
        return np.array([]), np.array([])

    # from utils.visualize import Plot
    # Plot.draw_pc_polar(pc_xyzrgb=pc, idx=0, center_idxes=candidate_center_idx, polar_masks=candidate_masks)
    height_center_idx, height_masks = _screen_mask_center(pc, candidate_center_idx, candidate_masks, mode='height')

    # Plot.draw_pc_polar(pc_xyzrgb=pc, idx=1, center_idxes=height_center_idx, polar_masks=height_masks)
    height_center_idx, height_masks, height_independent_idx = _separate_mask(pc, height_center_idx, height_masks, mode='area')

    # Plot.draw_pc_polar(pc_xyzrgb=pc, idx=2, center_idxes=height_center_idx, polar_masks=height_masks)
    height_masks = _apex_masks(pc, pre_masks, height_center_idx, height_masks, height_independent_idx)

    # Plot.draw_pc_polar(pc_xyzrgb=pc, idx=3, center_idxes=height_center_idx, polar_masks=height_masks)
    final_center_idx = height_center_idx
    final_masks = height_masks
    return final_center_idx, final_masks


def get_point_instance(points, center_idxes, polar_masks):
    polar_masks = np.array(polar_masks)
    point_instance = np.ones(len(points)) * -1
    for i, idx in enumerate(center_idxes):
        in_mask = points_in_mask(points, points[idx], polar_masks[i])
        point_instance[in_mask] = i
    return point_instance


def _support_to_mask(support_points, center, ray_num):
    if support_points is None:
        return None
    angles = np.arange(-np.pi, np.pi, np.pi * 2 / ray_num)
    support_points -= center[:2]
    support_points_len = np.sqrt(np.sum(support_points[:, :2] ** 2, axis=1))
    support_point_angles = np.zeros(len(support_points))

    for j, point in enumerate(support_points):
        if support_points_len[j] == 0:
            same_rt_points = support_points[(support_points[:, 0] == 0) * (support_points[:, 1] != 0)]
            if len(same_rt_points) != 0 and same_rt_points[0][1] > 0:
                support_point_angles[j] = - np.pi / 2
            else:
                support_point_angles[j] = np.pi / 2  # both direct is ok on single point
        else:
            support_point_angles[j] = np.arctan2(point[1], point[0])
    polar_mask = np.zeros(ray_num)
    for j, angle in enumerate(angles):
        angle_errors = support_point_angles - angle
        pos_angle_errors = angle_errors[angle_errors >= 0]
        neg_angle_errors = angle_errors[angle_errors < 0]
        lower_idx = -1
        higher_idx = -1
        lower_angle = -1
        higher_angle = -1
        if len(pos_angle_errors) != 0:
            higher_idx = (angle_errors >= 0).nonzero()[0][np.argmin(pos_angle_errors)]
            higher_angle = angle_errors[higher_idx]
        if len(neg_angle_errors) != 0:
            lower_idx = (angle_errors < 0).nonzero()[0][np.argmax(neg_angle_errors)]
            lower_angle = - angle_errors[lower_idx]
        if lower_idx == -1:
            lower_idx = (angle_errors >= 0).nonzero()[0][np.argmax(pos_angle_errors)]
            lower_angle = 2 * np.pi - angle_errors[lower_idx]
        if higher_idx == -1:
            higher_idx = (angle_errors < 0).nonzero()[0][np.argmin(neg_angle_errors)]
            higher_angle = 2 * np.pi + angle_errors[higher_idx]
        higher_len = support_points_len[higher_idx]
        lower_len = support_points_len[lower_idx]
        higher_angle = np.maximum(higher_angle, 0)
        lower_angle = np.maximum(lower_angle, 0)
        if lower_angle + higher_angle < np.pi:
            angle_len = _calc_angle_len(higher_angle, lower_angle, higher_len, lower_len)
        else:
            angle_len = 0
        polar_mask[j] = angle_len
    return polar_mask


def _get_intersection_point(point_1_1, point_1_2, point_2_1, point_2_2):
    k_1, b_1 = _get_slope_intercept(point_1_1[0], point_1_1[1], point_1_2[0], point_1_2[1])
    k_2, b_2 = _get_slope_intercept(point_2_1[0], point_2_1[1], point_2_2[0], point_2_2[1])
    k, b = _get_slope_intercept(k_1, b_1, k_2, b_2)
    x = -k
    y = b
    return np.array([x, y], dtype=np.float32)


def _get_slope_intercept(x_1, y_1, x_2, y_2):
    k = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - k * x_1
    return k, b


def _get_mask_support_points(center, mask):
    ray_num = mask.shape[-1]
    unit_angle = np.pi * 2 / ray_num
    angles = unit_angle * np.arange(0, ray_num) - np.pi
    mask_points = (mask * np.vstack((np.cos(angles), np.sin(angles)))).transpose() + center[:2]
    return mask_points


def _get_apex_idx(pc, center, mask):
    in_mask_idx = points_in_mask(pc, center, mask).nonzero()[0]
    apex_idx = in_mask_idx[np.argmax(pc[in_mask_idx, 2])]
    return apex_idx


def _get_point_dist(pc, idx_1, idx_2):
    return np.sqrt(np.sum((pc[idx_1, :2] - pc[idx_2, :2]) ** 2, axis=-1))


def _check_center_idx(pc, center, mask, cross_idx, factor=1.2):
    in_mask_idx = points_in_mask(pc, center, mask).nonzero()[0]
    if len(in_mask_idx) < 10:
        return False

    ray_num = mask.shape[-1]
    envelope_mask = _support_to_mask(_get_envelope(pc[in_mask_idx], center), center, ray_num)
    edge_idx = in_mask_idx[(~points_in_mask(pc[in_mask_idx], center, envelope_mask * 0.9)).nonzero()[0]]
    apex_idx = edge_idx[np.argmax(pc[edge_idx, 2])]
    apex_height = pc[apex_idx, 2]

    dot_product = np.sum((pc[in_mask_idx, :2] - pc[apex_idx, :2]) * (center - pc[apex_idx, :2]), axis=-1)
    foot_idx = np.argmax(dot_product)
    foot_height = pc[foot_idx, 2]

    # find center neighbor to avoid outlier
    nearest_idx = np.argpartition(np.sum((pc[in_mask_idx, :2] - center) ** 2, axis=-1), 8)[1:9]
    neigh_idx = in_mask_idx[nearest_idx[np.argmax(pc[in_mask_idx[nearest_idx], 2])]]
    center_height = pc[neigh_idx, 2]
    dist_to_apex = _get_point_dist(pc, neigh_idx, apex_idx)
    dist_to_foot = _get_point_dist(pc, neigh_idx, foot_idx)
    theo_height = (apex_height * dist_to_foot + foot_height * dist_to_apex) / (dist_to_foot + dist_to_apex)
    min_height = np.minimum(apex_height, foot_height)
    ratio = (center_height - min_height) / (theo_height - min_height)
    print(ratio)
    return ratio > factor


def _smooth_masks(masks):
    smooth_masks = np.zeros(masks.shape)
    for i in range(masks.shape[-1]):
        left = masks[:, i - 1]
        mid = masks[:, i]
        if i == masks.shape[-1] - 1:
            right = masks[:, 0]
        else:
            right = masks[:, i + 1]
        smooth_masks[:, i] = (left + mid + right) / 3
    return smooth_masks


def _screen_mask_ratio(center_idx, masks, threshold=2.5):
    ratio = np.zeros((masks.shape[1], masks.shape[0]))
    for i in range(masks.shape[-1]):
        ratio_1 = masks[:, i - 1] / masks[:, i]
        if i == masks.shape[-1] - 1:
            ratio_2 = masks[:, 0] / masks[:, i]
        else:
            ratio_2 = masks[:, i + 1] / masks[:, i]
        ratio[i] = np.maximum(ratio_1, ratio_2)
    ratio = np.max(ratio, axis=0)
    rational = ratio < threshold
    final_center_idx = center_idx[rational]
    final_masks = masks[rational]
    return final_center_idx, final_masks


def _screen_mask_center(pc, center_idx, masks, mode='height'):
    final_center_idx = []
    final_masks = []

    if mode == 'height':
        descend_idx = np.argsort(-pc[center_idx, 2])
    elif mode == 'area':
        mask_relative_area = np.sum(masks ** 2, axis=-1)
        descend_idx = np.argsort(-mask_relative_area)
    else:
        descend_idx = np.argsort(-pc[center_idx, 2])

    for i in descend_idx:
        if len(final_center_idx) > 0:
            final_in_tmp = points_in_mask(pc[final_center_idx], pc[center_idx[i]], masks[i])
            duplicate_in_final = final_in_tmp * (np.abs(pc[center_idx[i]][1] - pc[final_center_idx][:, 1]) < 1.0)
            if np.sum(duplicate_in_final) > 0:
                continue
        final_center_idx += [center_idx[i]]
        final_masks += [masks[i]]

    return np.array(final_center_idx), np.array(final_masks)


def _separate_mask(pc, center_idx, masks, mode='height'):
    ray_num = masks.shape[-1]
    final_center_idx = []
    final_masks = []
    final_mask_points = []
    is_independent = []

    if mode == 'height':
        descend_idx = np.argsort(-pc[center_idx, 2])
    elif mode == 'area':
        mask_relative_area = np.sum(masks ** 2, axis=-1)
        descend_idx = np.argsort(-mask_relative_area)
    else:
        descend_idx = np.argsort(-pc[center_idx, 2])

    for i in descend_idx:
        center_1 = pc[center_idx[i], :2]
        mask_1 = masks[i]
        mask_points_1 = _get_mask_support_points(center_1, mask_1)

        insert_1 = True
        independent = True
        combine_idx = -1
        for j in range(len(final_center_idx)):
            center_2 = pc[final_center_idx[j], :2]
            mask_2 = final_masks[j]
            mask_points_2 = final_mask_points[j]

            start_idx_1, end_idx_1, start_idx_2, end_idx_2 = _get_cross_idx(center_1, mask_1, center_2, mask_2)

            # check 1 center overlap
            if start_idx_1 is None:
                in_mask = points_in_mask(center_1.reshape(1, len(center_1)), center_2, mask_2)
                if in_mask:
                    insert_1 = False
                    break
                else:
                    continue

            independent = False
            cross_1 = np.zeros(ray_num)
            cross_2 = np.zeros(ray_num)
            cross_1[np.arange(start_idx_1, end_idx_1 + 1)] = 1
            cross_2[np.arange(start_idx_2, end_idx_2 + 1)] = 1
            ray_threshold = int(ray_num * 0.6)

            # check 2 edge overlap
            if np.sum(cross_1 == 0) < ray_threshold or np.sum(cross_2 == 0) < ray_threshold:
                insert_1 = False
                break

            center_direct = center_1 - center_2
            center_dist = np.sqrt(np.sum(center_direct ** 2))

            # check 3 center dist
            if center_dist < 0.5:
                support_points_2 = _get_envelope(np.concatenate((mask_points_1, mask_points_2), axis=0), center_2)
                final_masks[j] = _support_to_mask(support_points_2, center_2, ray_num)
                final_mask_points[j] = _get_mask_support_points(center_2, final_masks[j])
                is_independent[j] = False

                insert_1 = False
                combine_idx = j
                center_1 = center_2
                mask_1 = final_masks[j]
                mask_points_1 = final_mask_points[j]
                continue

            if end_idx_1 == ray_num - 1:
                end_idx_1 = -1
            if end_idx_2 == ray_num - 1:
                end_idx_2 = -1
            intersection_point_1 = _get_intersection_point(mask_points_1[start_idx_1],
                                                           mask_points_1[start_idx_1 - 1],
                                                           mask_points_2[end_idx_2], mask_points_2[end_idx_2 + 1])
            intersection_point_2 = _get_intersection_point(mask_points_1[end_idx_1], mask_points_1[end_idx_1 + 1],
                                                           mask_points_2[start_idx_2],
                                                           mask_points_2[start_idx_2 - 1])
            intersection_points = np.vstack((intersection_point_1, intersection_point_2))
            support_points_1 = np.concatenate((mask_points_1[cross_1 == 0], intersection_points))
            support_points_2 = np.concatenate((mask_points_2[cross_2 == 0], intersection_points))
            new_mask_1 = _support_to_mask(support_points_1, center_1, ray_num)
            new_mask_2 = _support_to_mask(support_points_2, center_2, ray_num)

            # check 4 peak shape
            # reconfirm_peak_1 = _check_center_idx(pc, center_1, new_mask_1, int((start_idx_1 + end_idx_1) / 2))
            # if not reconfirm_peak_1:
            #     support_points_2 = get_envelope(np.concatenate((mask_points_1, mask_points_2), axis=0), center_2)
            #     final_masks[j] = _support_to_mask(support_points_2, center_2, ray_num)
            #     final_mask_points[j] = _get_mask_support_points(center_2, final_masks[j])
            #     is_independent[j] = False
            #
            #     insert_1 = False
            #     combine_idx = j
            #     center_1 = center_2
            #     mask_1 = final_masks[j]
            #     mask_points_1 = final_mask_points[j]
            #     continue

            # check 5 separate status
            sum_old_1 = np.sum(mask_1 ** 2)
            sum_old_2 = np.sum(mask_2 ** 2)
            sum_new_1 = np.sum(new_mask_1 ** 2)
            sum_new_2 = np.sum(new_mask_2 ** 2)
            ratio_1 = sum_new_1 / sum_old_1
            ratio_2 = sum_new_2 / sum_old_2
            if ratio_1 < 0.5:
                if ratio_2 < 0.7:
                    support_points_2 = _get_envelope(np.concatenate((mask_points_1, mask_points_2), axis=0), center_2)
                    final_masks[j] = _support_to_mask(support_points_2, center_2, ray_num)
                    final_mask_points[j] = _get_mask_support_points(center_2, final_masks[j])
                    is_independent[j] = False

                    insert_1 = False
                    combine_idx = j
                    center_1 = center_2
                    mask_1 = final_masks[j]
                    mask_points_1 = final_mask_points[j]
                    continue
                else:
                    insert_1 = False
                    break

            mask_1 = new_mask_1
            mask_points_1 = _get_mask_support_points(center_1, mask_1)
            final_masks[j] = new_mask_2
            final_mask_points[j] = _get_mask_support_points(center_2, final_masks[j])
            is_independent[j] = False
        # add 1 to final
        if insert_1:
            final_center_idx += [center_idx[i]]
            final_masks += [mask_1]
            final_mask_points += [mask_points_1]
            is_independent += [independent]

        if combine_idx != -1:
            final_masks[combine_idx] = mask_1
            final_mask_points[combine_idx] = mask_points_1

    return np.array(final_center_idx), np.array(final_masks), np.array(is_independent).nonzero()[0]


def _apex_masks(pc, pre_masks, final_center_idx, final_masks, independent_idx):
    if len(independent_idx) == 0:
        return final_masks
    masks = pre_masks[independent_idx]
    for i, mask in enumerate(masks):
        in_mask_idx = points_in_mask(pc, pc[final_center_idx[independent_idx[i]]], mask).nonzero()[0]
        if len(in_mask_idx) == 0:
            continue
        points = pc[in_mask_idx]
        apex_idx = in_mask_idx[np.argmax(points[:, 2])]
        if apex_idx != final_center_idx[independent_idx[i]]:
            apex_mask = pre_masks[apex_idx]
            apex_mask_points = _get_mask_support_points(pc[apex_idx], apex_mask)
            intersect = False
            for j in range(len(final_center_idx)):
                center = pc[final_center_idx[j]]
                if np.sum(points_in_mask(apex_mask_points, center, final_masks[j])) > 0:
                    intersect = True
                    break
            if intersect:
                continue
            final_masks[independent_idx[i]] = apex_mask
    return final_masks


def _get_cross_idx(center_1, mask_1, center_2, mask_2):
    ray_num = mask_1.shape[-1]
    unit_angle = np.pi * 2 / ray_num

    mask_points_1 = _get_mask_support_points(center_1, mask_1)
    mask_points_2 = _get_mask_support_points(center_2, mask_2)
    points_1_in_2 = points_in_mask(mask_points_1, center_2, mask_2)
    points_2_in_1 = points_in_mask(mask_points_2, center_1, mask_1)
    if np.sum(points_1_in_2) < 2 or np.sum(points_2_in_1) < 2:
        return None, None, None, None

    relative_direct = center_2 - center_1
    relative_angle = np.arctan2(relative_direct[1], relative_direct[0])
    lower_idx = np.floor(relative_angle / unit_angle).astype(np.int) + int(ray_num / 2)
    if lower_idx == -1:
        lower_idx = ray_num - 1
    if lower_idx == ray_num:
        lower_idx = 0
    higher_idx = lower_idx + 1
    if higher_idx == ray_num:
        higher_idx = 0

    # find widest intersection
    start_idx_1 = higher_idx - int(ray_num / 2)
    end_idx_1 = lower_idx - int(ray_num / 2)
    if start_idx_1 < 0:
        start_idx_1 += ray_num
    if end_idx_1 < 0:
        end_idx_1 += ray_num
    if start_idx_1 != 0:
        start_idx_1 -= ray_num
    for i in range(ray_num):
        if points_1_in_2[start_idx_1 + i]:
            start_idx_1 += i
            break
    for i in range(ray_num):
        if points_1_in_2[end_idx_1 - i]:
            end_idx_1 -= i
            break

    start_idx_2 = higher_idx
    end_idx_2 = lower_idx
    if start_idx_2 != 0:
        start_idx_2 -= ray_num
    for i in range(ray_num):
        if points_2_in_1[start_idx_2 + i]:
            start_idx_2 += i
            break
    for i in range(ray_num):
        if points_2_in_1[end_idx_2 - i]:
            end_idx_2 -= i
            break
    return start_idx_1, end_idx_1, start_idx_2, end_idx_2


def _calc_angle_len(angle_a, angle_b, len_a, len_b):
    if angle_a == 0:
        return len_a
    if angle_b == 0:
        return len_b
    numerator = len_a * len_b * (np.sin(angle_a) * np.cos(angle_b) + np.sin(angle_b) * np.cos(angle_a))
    denominator = len_a * np.sin(angle_a) + len_b * np.sin(angle_b)
    return numerator / denominator


def _get_support_points(instance_points):
    unique_rts = np.unique(instance_points[:, 0])
    support_points_higher = np.zeros((len(unique_rts), 2))
    support_points_lower = np.zeros((len(unique_rts), 2))
    for j, rt in enumerate(unique_rts):
        temp_rt_points = instance_points[instance_points[:, 0] == rt]
        min_idx = np.argmin(temp_rt_points[:, 1])
        max_idx = np.argmax(temp_rt_points[:, 1])
        support_points_higher[j] = temp_rt_points[max_idx, :2]
        support_points_lower[j] = temp_rt_points[min_idx, :2]
    support_points = _get_side_support_points(support_points_higher, 1)
    support_points += _get_side_support_points(support_points_lower, -1)
    return np.array(support_points)


def _get_envelope(instance_points, center):
    # a: border to center
    # b: border to other
    # c: other to center

    points_to_center = center[:2] - instance_points[:, :2]
    dist2 = np.sum(points_to_center ** 2, axis=1)
    zero_idx = dist2 < 0.0001
    dist2[zero_idx] = 0.0001
    dist = np.sqrt(dist2)
    first_border_idx = np.argmax(dist)
    tmp_border_idx = first_border_idx
    envelope = [instance_points[first_border_idx, :2]]

    while True:
        border_to_points = instance_points[:, :2] - instance_points[tmp_border_idx, :2]
        cross = np.cross(points_to_center[tmp_border_idx], border_to_points)
        positive_idx = ((cross > 0) * (dist2 > 0.0001)).nonzero()[0]
        if len(positive_idx) == 0:
            envelope += [center[:2]]
            c2 = np.sum(border_to_points ** 2, axis=1)
            cos_theta = (dist2[tmp_border_idx] + dist2 - c2) / (2 * dist[tmp_border_idx] * dist)
            cos_theta[zero_idx] = 1
            next_border_idx = np.argmin(cos_theta)
        else:
            b2 = np.sum(border_to_points ** 2, axis=1)
            b = np.sqrt(b2)
            # (a2 + b2 - c2) / (2 * a * b)
            cos_theta = (dist2[tmp_border_idx] + b2 - dist2) / (2 * dist[tmp_border_idx] * b + 1e-6)
            next_border_idx = positive_idx[np.argmin(cos_theta[positive_idx])]

        if next_border_idx == first_border_idx:
            break
        tmp_border_idx = next_border_idx
        envelope += [instance_points[tmp_border_idx, :2]]

    return np.array(envelope)


def _get_side_support_points(points, direction):
    support_points = []
    sort_idx = np.argsort(points[:, 1])[::-direction]
    apex_idx = sort_idx[0]
    left_idx = apex_idx
    right_idx = apex_idx
    for j in range(0, len(sort_idx)):
        temp_idx = sort_idx[j]
        if temp_idx < apex_idx:
            if temp_idx < left_idx:
                support_points += [points[temp_idx]]
                left_idx = temp_idx
            elif points[temp_idx, 1] == points[left_idx, 1]:
                support_points += [points[temp_idx]]
        else:
            if temp_idx > right_idx:
                support_points += [points[temp_idx]]
                right_idx = temp_idx
            elif points[temp_idx, 1] == points[right_idx, 1]:
                support_points += [points[temp_idx]]
    return support_points


def _sample_from_gaussian(distances, heatmap, radius):
    sigma = radius / 3
    power_coef = -1 / (2 * np.power(sigma, 2))
    heatmap[distances <= radius] = np.maximum(heatmap[distances <= radius], np.exp(distances[distances <= radius] ** 2 * power_coef))
    return heatmap
