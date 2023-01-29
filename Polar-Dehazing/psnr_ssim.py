import glob

import torch
import torch.nn.functional as F
from math import log10
from skimage import metrics
from PIL import Image
import torchvision.transforms as transforms


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    # 4维
    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]

    # 3维
    # dehaze_list_np = [dehaze_list[ind].permute(1, 2, 0).data.cpu().numpy() for ind in range(len(dehaze_list))]
    # gt_list_np = [gt_list[ind].permute(1, 2, 0).data.cpu().numpy() for ind in range(len(dehaze_list))]

    ssim_list = [metrics.structural_similarity(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def psnr_ssim(result_path, gt_path):
    psnr_list = []
    ssim_list = []
    gt_list = sorted(glob.glob(gt_path + '/*'))
    result_list = sorted(glob.glob(result_path + '/*'))
    for i in range(len(gt_list)):
        gt = Image.open(gt_list[i])
        result = Image.open(result_list[i])
        gt = transforms.ToTensor()(gt).unsqueeze(0)
        result = transforms.ToTensor()(result).unsqueeze(0)
        psnr_list.extend(to_psnr(result, gt))
        ssim_list.extend(to_ssim_skimage(result, gt))
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim