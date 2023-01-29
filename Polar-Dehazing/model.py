import torch
import torch.nn as nn
import torch.nn.functional as F

from Oct3d import *


class FC(nn.Module):

    def forward(self, x0, x1, x2, x3, x4):
        x0 = torch.max(x0, 2)[0]
        x1 = torch.max(x1, 2)[0]
        x2 = torch.max(x2, 2)[0]
        x3 = torch.max(x3, 2)[0]
        x4 = torch.max(x4, 2)[0]
        return x0, x1, x2, x3, x4


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.InstanceNorm3d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.InstanceNorm3d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.InstanceNorm3d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.InstanceNorm3d(512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.InstanceNorm3d(512),
            nn.ReLU()
        )

    def forward(self, haze_0, haze_45, haze_90, haze_135):
        haze_0 = haze_0.unsqueeze(0)
        haze_45 = haze_45.unsqueeze(0)
        haze_90 = haze_90.unsqueeze(0)
        haze_135 = haze_135.unsqueeze(0)
        x = torch.cat((haze_0, haze_45, haze_90, haze_135), 0)
        x = x.permute(1, 2, 0, 3, 4)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x1, x2, x3, x4, x5


class Oct3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Oct3D, self).__init__()
        self.layer1 = FirstOctaveCBR(in_channel, out_channel, kernel_size=(1, 1, 1), stride=1, padding=0, alpha=0.5)
        self.layer2 = OctaveCBR(out_channel, out_channel, kernel_size=(3, 3, 3), stride=1, padding=1, alpha=0.5)
        self.layer3 = LastOCtaveCBR(out_channel, out_channel, kernel_size=(3, 3, 3), stride=1, padding=1, alpha=0.5)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3


class OctNet(nn.Module):
    def __init__(self):
        super(OctNet, self).__init__()
        self.oct0 = Oct3D(64, 64)
        self.oct1 = Oct3D(128, 128)
        self.oct2 = Oct3D(256, 256)
        self.oct3 = Oct3D(512, 512)
        self.oct4 = Oct3D(512, 512)

    def forward(self, x0, x1, x2, x3, x4):
        x_oc0 = self.oct0(x0)
        x_oc1 = self.oct1(x1)
        x_oc2 = self.oct2(x2)
        x_oc3 = self.oct3(x3)
        x_oc4 = self.oct4(x4)
        return x_oc0, x_oc1, x_oc2, x_oc3, x_oc4


class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3, x4, x5):
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = self.layer1(x5)

        x4_6 = torch.cat((x4, x6), 1)
        x4_6 = F.interpolate(x4_6, scale_factor=2, mode='bilinear', align_corners=False)
        x7 = self.layer2(x4_6)

        x3_7 = torch.cat((x3, x7), 1)
        x3_7 = F.interpolate(x3_7, scale_factor=2, mode='bilinear', align_corners=False)
        x8 = self.layer3(x3_7)

        x2_8 = torch.cat((x2, x8), 1)
        x2_8 = F.interpolate(x2_8, scale_factor=2, mode='bilinear', align_corners=False)
        x9 = self.layer4(x2_8)

        x1_9 = torch.cat((x1, x9), 1)
        x10 = self.layer5(x1_9)
        return x10


class PolNet(nn.Module):
    def __init__(self):
        super(PolNet, self).__init__()
        self.fn = FeatureNet()
        self.oct = OctNet()
        self.fc = FC()
        self.decoder = Deconv()

    def forward(self, haze_0, haze_45, haze_90, haze_135):
        x1, x2, x3, x4, x5 = self.fn(haze_0, haze_45, haze_90, haze_135)
        x_oc0, x_oc1, x_oc2, x_oc3, x_oc4 = self.oct(x1, x2, x3, x4, x5)
        x_f0, x_f1, x_f2, x_f3, x_f4 = self.fc(x_oc0, x_oc1, x_oc2, x_oc3, x_oc4)
        y = self.decoder(x_f0, x_f1, x_f2, x_f3, x_f4)
        return y


if __name__ == "__main__":
    x1 = torch.rand((1, 3, 480, 992))
    x2 = torch.rand((1, 3, 480, 992))
    x3 = torch.rand((1, 3, 480, 992))
    x4 = torch.rand((1, 3, 480, 992))
    net = PolNet()
    out = net(x1, x2, x3, x4)
    print(out.shape)