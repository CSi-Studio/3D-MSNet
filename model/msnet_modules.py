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
import torch.nn as nn
import msnet_utils
import pytorch_utils as pt_utils
import torch.nn.functional as F
from typing import List


# Local Spatial Encoding
class LocalSpatialEncodingModule(nn.Module):
    def __init__(self, *, mlp: List[int], radius: float = None, nsample: int = None, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()
        mlp[0] += 3
        self.mlps = pt_utils.SharedMLP(mlp, bn=False, instance_norm=False)
        self.radius = radius
        self.nsample = nsample
        self.pool_method = pool_method

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :return:
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        idx = msnet_utils.ms_query(self.radius, self.nsample, xyz, xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = msnet_utils.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= xyz.transpose(1, 2).unsqueeze(-1)
        unique_cnt = torch.sum(idx - idx[:, :, 0:1] != 0, dim=2) + 1

        features = self.mlps(grouped_xyz)  # (B, mlp[-1], npoint, nsample)

        if self.pool_method == 'max_pool':
            features = F.max_pool2d(features, kernel_size=[1, features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pool_method == 'avg_pool':
            features = F.avg_pool2d(features, kernel_size=[1, features.size(3)])  # (B, mlp[-1], npoint, 1)
        else:
            raise NotImplementedError

        features = features.squeeze(-1).transpose(1, 2)  # (B, mlp[-1], npoint)
        density = unique_cnt / float(features.shape[-1])
        features = torch.cat((features, density.unsqueeze(-1)), dim=-1).contiguous()

        return features


# Set Abstraction, Encoding block
class MSNetSAModule(nn.Module):

    def __init__(self, *, use_sample: bool, in_channel: int, out_channel: int, mlps: List[int], grid_x: int, grid_y: int,
                 l_x: List[float], l_y: List[float]):

        super().__init__()

        assert len(l_x) == len(l_y)

        self.use_sample = use_sample
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.l_x = l_x
        self.l_y = l_y

        self.k = (2 * grid_x + 1) * (2 * grid_y + 1)
        self.conv = pt_utils.Conv2d(self.in_channel + 2, self.out_channel, kernel_size=(1, self.k))
        self.mlps = pt_utils.SharedMLP(mlps, bn=False) if (mlps is not None) else None

    def forward(self, xyz: torch.Tensor, feature: torch.Tensor = None, new_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param feature: (B, N, C) tensor of the descriptors of the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        feature = feature.contiguous()
        if new_xyz is None:
            if self.use_sample:
                score = (xyz[:, :, 2].transpose(0, 1) / torch.max(xyz[:, :, 2], dim=1)[0]).transpose(0, 1)
                random = torch.rand(score.shape, dtype=torch.float32).cuda() * 1.1
                residual = random - score
                n_sample = int(max(1024, min(10000, xyz.shape[1] / 2)))
                sample_idx = torch.argsort(residual, dim=-1)[:, :n_sample]
                sample_idx = sample_idx.type(torch.int)
                new_xyz = msnet_utils.gather_operation(xyz_flipped, sample_idx).transpose(1, 2).contiguous()
            else:
                if xyz.shape[1] > 10000:
                    new_xyz = msnet_utils.gather_operation(xyz_flipped, msnet_utils.furthest_point_sample(xyz, 10000))\
                        .transpose(1, 2).contiguous()
                else:
                    new_xyz = xyz

        result_feature = []
        for i in range(len(self.l_x)):
            """
                idx: (B, M, K, 100)
                weight: (B, M, K, 100)
                new_feature: (B, M, K, C)
                result_feature: (B, M, C')
            """
            idx, weight = msnet_utils.interpolate_nn(xyz, new_xyz, self.grid_x, self.grid_y, self.l_x[i], self.l_y[i])
            xyz_feature = torch.cat((xyz[:, :, :2], feature), dim=-1)
            new_feature = msnet_utils.bilinear_interpolate(xyz_feature, idx, weight)

            center = torch.cat((new_xyz[:, :, :2], torch.zeros(new_xyz.shape[0], new_xyz.shape[1], new_feature.shape[-1] - 2).cuda()), dim=-1)
            center = center.unsqueeze(2).repeat(1, 1, new_feature.shape[-2], 1)
            new_feature -= center
            result_feature.append(self.conv(new_feature.permute(0, 3, 1, 2)).squeeze(-1))

        result_feature = torch.cat(result_feature, dim=1).unsqueeze(-1)
        if self.mlps is not None:
            result_feature = self.mlps(result_feature).squeeze(-1).permute(0, 2, 1)

        return new_xyz, result_feature


# Feature propagation, Decoding block
class MSNetFPModule(nn.Module):

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        known_feats = known_feats.permute(0, 2, 1).contiguous()
        unknow_feats = unknow_feats.permute(0, 2, 1)
        if known is not None:
            dist, idx = msnet_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm * (torch.max(dist, dim=2, keepdim=True)[0] < 1)

            interpolated_feats = msnet_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1).permute(0, 2, 1)
