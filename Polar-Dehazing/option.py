import os, argparse
import torch, warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--datasets', type=str, default='../datasets')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir', type=str, default='./checkpoints')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--model_epoch', type=int, default=197, help='Takes effect when using --resume ')
parser.add_argument('--crop_size', type=int, default=[256, 256])


opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(opt)
