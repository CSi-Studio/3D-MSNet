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
import torch.optim as optim
import os
from model.msnet_model import backbone_msnet, sem_net, center_net, polar_mask_net,\
    SemanticLoss, MaskIOULoss, CenterLoss, semantic_accuracy, mask_accuracy, center_accuracy
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MsNet:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backbone = backbone_msnet().cuda()
        self.sem_net = sem_net().cuda()
        self.polar_mask_net = polar_mask_net().cuda()
        self.center_net = center_net().cuda()
        self.init_optimizer()

    def init_optimizer(self):
        optim_params = [
            {'params': self.backbone.parameters(), 'lr': self.cfg.learning_rate[0], 'betas': (0.9, 0.999), 'eps': 1e-08},
            {'params': self.sem_net.parameters(), 'lr': self.cfg.learning_rate[1], 'betas': (0.9, 0.999), 'eps': 1e-08},
            {'params': self.center_net.parameters(), 'lr': self.cfg.learning_rate[2], 'betas': (0.9, 0.999), 'eps': 1e-08},
            {'params': self.polar_mask_net.parameters(), 'lr': self.cfg.learning_rate[3], 'betas': (0.9, 0.999), 'eps': 1e-08}
        ]

        if self.cfg.optimizer == 'Adam':
            self.optimizer = optim.Adam(optim_params, weight_decay=0.01)
        else:
            self.optimizer = optim.SGD(optim_params)

    def run(self, data, is_train):
        if is_train:
            self.optimizer.zero_grad()
            self.backbone.train()
            self.center_net.train()
            self.sem_net.train()
            self.polar_mask_net.train()
        else:
            self.backbone.eval()
            self.center_net.eval()
            self.sem_net.eval()
            self.polar_mask_net.eval()

        bat_pc, bat_ins, bat_center_idx, bat_pmask, bat_center_heatmap = data
        bat_pc, bat_ins, bat_center_idx, bat_pmask, bat_center_heatmap = \
            bat_pc.cuda(), bat_ins.cuda(), bat_center_idx.cuda(), bat_pmask.cuda(), bat_center_heatmap.cuda()

        """ feature extraction """
        point_features = self.backbone(bat_pc[:, :, 0:3])
        valid_center_idx = (bat_center_idx >= 0).nonzero()

        """ semantic segmentation """
        sems = self.sem_net(point_features)
        gt_sems = (bat_ins != -1).long()
        semantic_loss = SemanticLoss(alpha=0.5, gamma=2)
        sem_loss = semantic_loss(sems, gt_sems)

        """ center prediction """
        pre_center = self.center_net(point_features).squeeze(-1)
        gt_center = bat_center_heatmap
        center_loss = CenterLoss(gamma=2, beta=1)
        ct_loss = center_loss(pre_center, bat_ins, gt_center)

        """ mask prediction """
        pre_masks = self.polar_mask_net(point_features)
        center_masks = pre_masks[valid_center_idx[:, 0], bat_center_idx[bat_center_idx >= 0]]
        gt_masks = bat_pmask[bat_center_idx >= 0]
        mask_iou_loss = MaskIOULoss()
        mask_loss = mask_iou_loss(center_masks, gt_masks)

        total_loss = 60 * sem_loss + 45 * ct_loss + 3 * mask_loss

        ct_acc = center_accuracy(pre_center, bat_ins, gt_center)
        sem_acc = semantic_accuracy(sems, gt_sems)
        mask_acc = mask_accuracy(center_masks, gt_masks)

        if is_train:
            total_loss.backward()
            self.optimizer.step()

        # visualize
        use_visualize = False
        # visualize(bat_pc, bat_center_idx, center_masks, pre_center, sems)

        return total_loss, 60 * sem_loss, 45 * ct_loss, 3 * mask_loss, sem_acc, ct_acc, mask_acc


def visualize(bat_pc, bat_center_idx, center_masks, pre_center, sems):
    from utils.visualize import Plot
    idx = 0
    points = bat_pc[0].cpu().detach().numpy()
    center_idxes = bat_center_idx[0].cpu().detach().numpy()
    polar_masks = center_masks.cpu().detach().numpy()
    center_idxes = center_idxes[center_idxes >= 0]
    polar_masks = polar_masks[:len(center_idxes)]
    Plot.draw_pc_polar(pc_xyzrgb=points[:, :3], idx=idx, center_idxes=center_idxes, polar_masks=polar_masks)
    idx += 1

    # pred center heatmap
    points = bat_pc[0].cpu().detach().numpy()
    Plot.draw_pc_heatmap(pc_xyz=points[:, :3], idx=idx, heatmap=pre_center[0].cpu().detach().numpy())
    idx += 1

    # pred center
    points = bat_pc[0].cpu().detach().numpy()
    center_map = np.zeros(pre_center[0].shape)
    center_map[(pre_center[0].cpu().detach().numpy() > 0.5) * (sems[0].cpu().detach().numpy() > 0.4)] = 1
    Plot.draw_pc_heatmap(pc_xyz=points[:, :3], idx=idx, heatmap=center_map)
    idx += 1

    # sem heatmap
    points = bat_pc[0].cpu().detach().numpy()
    Plot.draw_pc_heatmap(pc_xyz=points[:, :3], idx=idx, heatmap=sems[0].cpu().detach().numpy())
    idx += 1
