import os
import torch.cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image

from model import PolNet
from datasets import HazeDataset
from psnr_ssim import to_psnr, to_ssim_skimage
from numpy import *

os.makedirs('test_result', exist_ok=True)
os.makedirs('real_result', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = PolNet().to(device)
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./结果/model.pth'))

test_dataset = HazeDataset('./datasets', mode='test')
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
)

with torch.no_grad():
    net.eval()
    ssims = []
    psnrs = []
    for i, img in enumerate(test_dataloader):
        haze_0 = img['haze_0'].to(device)
        haze_45 = img['haze_45'].to(device)
        haze_90 = img['haze_90'].to(device)
        haze_135 = img['haze_135'].to(device)
        clear = img['clear'].to(device)
        predict = net(haze_0, haze_45, haze_90, haze_135)
        ssim1 = to_ssim_skimage(predict, clear)
        psnr1 = to_psnr(predict, clear)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        # save image
        save_image(predict, './test_result/%s.jpg' % str(i+1), normalize=False)
    avr_psnr = sum(psnrs) / len(psnrs)
    avr_ssim = sum(ssims) / len(ssims)
print(f'ssim:{avr_ssim:.4f}| psnr:{avr_psnr:.4f}')
