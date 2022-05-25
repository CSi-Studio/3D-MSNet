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
import torch.nn as nn
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from model.msnet_modules import LocalSpatialEncodingModule, MSNetSAModule, MSNetFPModule


class backbone_msnet(nn.Module):

    def __init__(self):
        super(backbone_msnet, self).__init__()
        self.lse = LocalSpatialEncodingModule(mlp=[0, 16, 16], radius=0.3, nsample=16)
        self.sa1 = MSNetSAModule(use_sample=True, in_channel=17,  out_channel=32,  mlps=[32, 32],   grid_x=2, grid_y=2, l_x=[0.2], l_y=[0.2])
        self.sa2 = MSNetSAModule(use_sample=True, in_channel=32,  out_channel=32,  mlps=[32, 64],   grid_x=2, grid_y=2, l_x=[0.4], l_y=[0.4])
        self.sa3 = MSNetSAModule(use_sample=True, in_channel=64,  out_channel=64,  mlps=[64, 128],  grid_x=2, grid_y=2, l_x=[0.8], l_y=[0.8])
        self.sa4 = MSNetSAModule(use_sample=True, in_channel=128, out_channel=128, mlps=[128, 256], grid_x=2, grid_y=2, l_x=[1.6], l_y=[1.6])
        self.fp4 = MSNetFPModule(mlp=[384, 256])
        self.fp3 = MSNetFPModule(mlp=[320, 256])
        self.fp2 = MSNetFPModule(mlp=[288, 256])
        self.fp1 = MSNetFPModule(mlp=[273, 256, 256])

    def forward(self, xyz):

        xyz = xyz.contiguous()
        features = self.lse(xyz)
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)
        l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(xyz, l1_xyz, features, l1_features)

        return l0_features


class sem_net(nn.Module):
    def __init__(self):
        super(sem_net, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_features):
        b1 = F.leaky_relu(self.fc1(point_features))
        b2 = self.sigmoid(self.fc2(b1))
        features = b2.squeeze(-1)

        return features


class center_net(nn.Module):
    def __init__(self):
        super(center_net, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, point_features):
        b1 = F.leaky_relu(self.fc1(point_features), negative_slope=0.2)
        b1 = self.dropout(b1)
        b2 = self.sigmoid(self.fc2(b1))
        return b2


class polar_mask_net(nn.Module):
    def __init__(self):
        super(polar_mask_net, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 36)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, point_features):
        b1 = F.leaky_relu(self.fc1(point_features), negative_slope=0.2)
        b1 = self.dropout(b1)
        b2 = self.fc2(b1)
        polar_mask = b2
        return polar_mask


class SemanticLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(SemanticLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        focal_loss = -(targets >= 0.4).float() * self.alpha * ((1. - inputs) ** self.gamma) * torch.log(inputs + 1e-8) \
                 - (1. - (targets >= 0.4).float()) * (1. - self.alpha) * (inputs ** self.gamma) * torch.log(
            1. - inputs + 1e-8)

        sem_loss_0 = torch.mean(focal_loss[targets == 0])
        sem_loss_1 = torch.mean(focal_loss[targets == 1])
        sem_loss = (sem_loss_0 + sem_loss_1) / 2
        return sem_loss


class CenterLoss(nn.Module):
    def __init__(self, gamma=2, beta=1):
        super(CenterLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta

    def forward(self, inputs, labels, targets):

        index_0 = (labels == -1)
        index_x = (targets == 0) * (labels != -1)
        index_1 = (targets != 0)
        ct_loss_1 = - torch.mean(torch.pow(targets[index_1], self.beta) *
                                 torch.pow(torch.abs(targets[index_1] - inputs[index_1]), self.gamma) *
                                 torch.log(1 - torch.abs(targets[index_1] - inputs[index_1])))
        ct_loss_x = - torch.mean(torch.pow(inputs[index_x], self.gamma) * torch.log((1 - inputs[index_x] + 1e-8)))
        ct_loss_0 = - torch.mean(torch.pow(inputs[index_0], self.gamma) * torch.log((1 - inputs[index_0] + 1e-8)))
        ct_loss = (ct_loss_0 + ct_loss_x + ct_loss_1) / 3
        return ct_loss


class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, input, target):
        """
         :param input:  shape (B,N,36), N is nr_box
         :param target: shape (B,N,36)
         :return: loss
         """
        input = input.reshape(-1, 36)
        target = target.reshape(-1, 36)
        total = torch.stack([input, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        negative_idx = l_min < 0
        l_max[negative_idx] -= l_min[negative_idx]
        l_min[negative_idx] = 0

        max_sum_l2 = torch.sum(torch.pow(l_max, 2), dim=-1) + 1E-6
        min_sum_l2 = torch.sum(torch.pow(l_min, 2), dim=-1) + 1E-6
        loss = torch.log(max_sum_l2 / min_sum_l2)
        mask_loss = torch.mean(loss)
        return mask_loss


def semantic_accuracy(inputs, targets):
    error = torch.abs(inputs - targets)
    error_0 = torch.mean(error[targets == 0])
    error_1 = torch.mean(error[targets == 1])
    acc = 1 - (error_0 + error_1) / 2
    return acc


def center_accuracy(inputs, labels, targets):
    index_0 = (labels == -1)
    index_x = (targets == 0) * (labels != -1)
    index_1 = (targets != 0)
    error = torch.abs(inputs - targets)
    error_0 = torch.mean(error[index_0])
    error_1 = torch.mean(error[index_x])
    error_2 = torch.mean(error[index_1])
    acc = 1 - (error_0 + error_1 + error_2) / 3
    return acc


def mask_accuracy(input, target):
    input = input.reshape(-1, 36)
    target = target.reshape(-1, 36)
    total = torch.stack([input, target], -1)
    l_max = total.max(dim=2)[0]
    l_min = total.min(dim=2)[0]

    negative_idx = l_min < 0
    l_max[negative_idx] -= l_min[negative_idx]
    l_min[negative_idx] = 0

    acc = torch.mean(torch.sum(l_min, dim=-1) / torch.sum(l_max, dim=-1))
    return acc
