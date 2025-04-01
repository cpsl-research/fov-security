import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from avstack.config import MODELS


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@MODELS.register_module()
class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        p_dropout: float = 0.0,
        first_layer_channels: int = 64,
        n_layers: int = 4,
        bilinear: bool = False,
    ):
        """Instantiate the UNet model
        
        n_channels: number of input image channels
        n_classes: number of output classes (output channels)
        p_dropout: dropout probability
        first_layer_channels: model width
        n_layers
        """

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.dropout = nn.Dropout2d(p=p_dropout)

        flc = first_layer_channels

        # input
        self.inc = DoubleConv(n_channels, flc)

        # encoders
        self.downs = nn.ModuleList([
            Down(flc * 2**i, flc * 2**(i+1))
            if (i+1) < self.n_layers else
            Down(flc * 2**i, flc * 2**(i+1) // factor)
            for i in range(self.n_layers)
        ])
        # self.down1 = Down(flc, 2 * flc)
        # self.down2 = Down(2 * flc, 4 * flc)
        # self.down3 = Down(4 * flc, 8 * flc)
        # self.down4 = Down(8 * flc, 16 * flc // factor)

        # decoder
        self.ups = nn.ModuleList([
            Up(flc * 2**(self.n_layers-i), flc * 2**(self.n_layers-i-1)//factor, bilinear)
            if (i+1) < self.n_layers else
            Up(flc * 2**(self.n_layers-i), flc, bilinear)
            for i in range(self.n_layers)
        ])
        # self.up1 = Up(16 * flc, 8 * flc // factor, bilinear)
        # self.up2 = Up(8 * flc, 4 * flc // factor, bilinear)
        # self.up3 = Up(4 * flc, 2 * flc // factor, bilinear)
        # self.up4 = Up(2 * flc, flc, bilinear)

        # output
        self.outc = OutConv(flc, n_classes)

    def forward(self, x):
        xd = self.inc(x)

        # -- downsampling
        xds = [xd]
        for i_d, down in enumerate(self.downs):
            xd = down(xd)
            if i_d > 0:
                xd = self.dropout(xd)
            xds.append(xd)

        # -- upsampling
        xu = xds[-1]
        for i_u, up in enumerate(self.ups):
            xu = up(xu, xds[-(i_u+2)])
            if (i_u + 1) < len(self.ups):
                xu = self.dropout(xu)

        # -- output
        logits = self.outc(xu)
        return logits
    
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.dropout(self.down2(x2))
        # x4 = self.dropout(self.down3(x3))
        # x5 = self.dropout(self.down4(x4))
        # x = self.dropout(self.up1(x5, x4))
        # x = self.dropout(self.up2(x, x3))
        # x = self.dropout(self.up3(x, x2))
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.downs = [torch.utils.checkpoint(down) for down in self.downs]
        self.ups = [torch.utils.checkpoint(up) for up in self.ups]
        self.outc = torch.utils.checkpoint(self.outc)
        # self.down1 = torch.utils.checkpoint(self.down1)
        # self.down2 = torch.utils.checkpoint(self.down2)
        # self.down3 = torch.utils.checkpoint(self.down3)
        # self.down4 = torch.utils.checkpoint(self.down4)
        # self.up1 = torch.utils.checkpoint(self.up1)
        # self.up2 = torch.utils.checkpoint(self.up2)
        # self.up3 = torch.utils.checkpoint(self.up3)
        # self.up4 = torch.utils.checkpoint(self.up4)

    def enable_eval_dropout(self):
        for module in self.modules():
            if "Dropout" in type(module).__name__:
                module.train()

    def predict_mc_dropout(self, x, n_iters: int, threshold: float):
        """Perform MC dropout prediction"""
        self.enable_eval_dropout()
        probs = torch.concat(tuple(self(x) for _ in range(n_iters)))
        mask_mean = torch.mean(probs, dim=0).squeeze()
        mask_bin = (mask_mean > threshold).squeeze()
        mask_std = torch.std(probs, dim=0).squeeze()
        return mask_bin.detach(), mask_mean.detach(), mask_std.detach()

    def load_weights_subdir(self, weight_dir: str, epoch: int = -1):
        # get the path to the weights
        try:
            last_subdir = sorted(next(os.walk(weight_dir))[1])[-1]
        except Exception as e:
            print(f"Cannot find anyting in subdir {last_subdir}")
            raise e
        if epoch == -1:
            weight_subdir = os.path.join(weight_dir, last_subdir)
            try:
                all_weights = sorted(glob(os.path.join(weight_subdir, "epoch*")))
                weights_path = all_weights[-1]
            except Exception as e:
                print(f"Cannot find weights in {weight_subdir}")
                raise e
        else:
            epoch_str = f"epoch_{epoch}.pth"
            weights_path = os.path.join(weight_dir, last_subdir, epoch_str)

        # load weights
        print(f"loading model weights from {weights_path}")
        self.load_state_dict(torch.load(weights_path))
        self.eval()


@MODELS.register_module()
class UNetBinary(UNet):
    def forward(self, x):
        return torch.sigmoid(super().forward(x))
