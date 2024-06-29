#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import exp
import math
from lpipsPyTorch import lpips

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))


class HaLoss(nn.Module):
    def __init__(self, hparams, coef=1):
        super().__init__()
        self.coef = coef
        self.Annealing = ExponentialAnnealingWeight(max = hparams.maskrs_max, min = hparams.maskrs_min, k = hparams.maskrs_k)
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, hparams, targets, global_step, mask_unet=False):
        ret = {}
        if 'a_embedded' in inputs:
            # ret['enc_l'] = 0.1 * torch.square(inputs['a_embedded'].detach() - inputs['re_a_embedded']).mean()
            if 'a_embedded_random_rec' in inputs:
                ret['rec_a_random'] = torch.mean(torch.abs(inputs['a_embedded_random'].detach() - inputs['a_embedded_random_rec'])) * hparams.weightRecA
                ret['mode_seeking'] = hparams.weightMS * \
                    torch.mean(torch.abs(inputs['a_embedded'].detach() - inputs['a_embedded_random'].detach())) / \
                    (torch.mean(torch.abs(inputs['rgb_fine'].detach() - inputs['rgb_fine_random'])) + 1e-5)

        if 'out_mask' in inputs:
            if mask_unet:
                ret['r_ms'] = self.mask_regularize(inputs['out_mask'], self.Annealing.getWeight(global_step))
                ret['r_focus'] = self.mask_focus_digit(inputs['out_mask'], hparams.mask_rd)
            ret['f_l'] = (1.0 - hparams.lambda_dssim) * ((1 - inputs['out_mask']) * torch.abs((inputs['rgb_fine'] - targets))).mean() + \
                        hparams.lambda_dssim * (1.0 - ssim(inputs['rgb_fine']*(1 - inputs['out_mask']), targets*(1 - inputs['out_mask'])))
                        # 0.005 * lpips(inputs['rgb_fine'], targets, net_type='alex')
        else:
            ret['f_l'] = (1.0 - hparams.lambda_dssim) * torch.abs(inputs['rgb_fine'] - targets).mean() + \
                        hparams.lambda_dssim * (1.0 - ssim(inputs['rgb_fine'], targets))

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

    def mask_regularize(self, mask, size_delta):
        # # l2 regularize
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        return loss_focus_size
    
    def mask_focus_digit(self, mask, digit_delta):
        # # l2 regularize
        focus_epsilon = 0.02
        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_digit