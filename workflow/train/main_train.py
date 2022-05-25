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
import time
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from utils.config import cfg
from utils.log import logger
from model.main_msnet import MsNet
from workflow.train.dataset_loader import Dataset


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 100 epochs"""
    if epoch == 0:
        return

    if epoch % 100 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2.0


def train(ms_net, dataset):
    loss_list = []
    acc_list = []
    for i, data in enumerate(dataset.train_data_loader):
        total_loss, sem_loss, ct_loss, mask_loss, sem_acc, ct_acc, mask_acc = ms_net.run(data, is_train=True)
        loss_list.append([total_loss.item(), sem_loss.item(), ct_loss.item(), mask_loss.item()])
        acc_list.append([sem_acc.item(), ct_acc.item(), mask_acc.item()])
    loss_list_final = np.mean(loss_list, axis=0)
    acc_list_final = np.mean(acc_list, axis=0)
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                "Epoch %3d Iteration %3d (train) %.3f %.3f %.3f %.3f %.3f %.3f %.3f" %
                (epoch, i, loss_list_final[0], loss_list_final[1], loss_list_final[2], loss_list_final[3],
                 acc_list_final[0], acc_list_final[1], acc_list_final[2]))
    return loss_list_final, acc_list_final


def val(ms_net, dataset):
    loss_list = []
    acc_list = []
    for i, data in enumerate(dataset.val_data_loader):
        total_loss, sem_loss, ct_loss, mask_loss, sem_acc, ct_acc, mask_acc = ms_net.run(data, is_train=False)
        loss_list.append([total_loss.item(), sem_loss.item(), ct_loss.item(), mask_loss.item()])
        acc_list.append([sem_acc.item(), ct_acc.item(), mask_acc.item()])
    loss_list_final = np.mean(loss_list, axis=0)
    acc_list_final = np.mean(acc_list, axis=0)
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                "Epoch %3d Iteration %3d (val) %.3f %.3f %.3f %.3f %.3f %.3f %.3f" %
                (epoch, i, loss_list_final[0], loss_list_final[1], loss_list_final[2], loss_list_final[3],
                 acc_list_final[0], acc_list_final[1], acc_list_final[2]))
    return loss_list_final, acc_list_final


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    """ Backup Network """
    if not os.path.exists(cfg.exp_path):
        os.mkdir(cfg.exp_path)
    os.system('cp ' + ROOT_DIR + '/model/main_msnet.py %s' % cfg.exp_path)
    os.system('cp ' + ROOT_DIR + '/model/msnet_model.py %s' % cfg.exp_path)
    os.system('cp ' + ROOT_DIR + '/model/msnet_modules.py %s' % cfg.exp_path)

    """ Load Dataset """
    dataset_anno = Dataset(cfg=cfg)
    dataset_anno.train_anno_loader()
    dataset_anno.val_anno_loader()

    logger.info('Training samples: {}'.format(len(dataset_anno.train_files)))
    logger.info('Validation samples: {}'.format(len(dataset_anno.val_files)))

    net = MsNet(cfg=cfg)
    min_loss = 10

    for epoch in range(cfg.epochs):

        adjust_learning_rate(net.optimizer, epoch)

        """ Training """
        train_loss, train_acc = train(net, dataset_anno)

        """ Validation """
        val_loss, val_acc = val(net, dataset_anno)

        """ Model param saving """
        if epoch > 100 and val_loss[0] < min_loss:
            min_loss = val_loss[0]
            torch.save(net.backbone.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'backbone', epoch))
            torch.save(net.sem_net.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'sem_net', epoch))
            torch.save(net.center_net.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'box_center_net', epoch))
            torch.save(net.polar_mask_net.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'polar_mask_net', epoch))
        if epoch % 50 == 0:
            torch.save(net.backbone.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'backbone', epoch))
            torch.save(net.sem_net.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'sem_net', epoch))
            torch.save(net.center_net.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'box_center_net', epoch))
            torch.save(net.polar_mask_net.state_dict(), '%s/%s_%.3d.pth' % (cfg.exp_path, 'polar_mask_net', epoch))