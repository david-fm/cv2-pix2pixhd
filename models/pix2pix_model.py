"""Baseline Pix2Pix building blocks (U-Net generator + 70x70 PatchGAN).

Used by `task1_baseline_approach.ipynb`. Kept in the same `models/` package as
the Pix2PixHD code so both notebooks share a single home for network code.
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """Standard 8-down / 8-up U-Net for 256x256 inputs (Isola et al., 2017)."""

    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        self.d1 = UNetDown(in_channels, ngf,        normalize=False)
        self.d2 = UNetDown(ngf,         ngf * 2)
        self.d3 = UNetDown(ngf * 2,     ngf * 4)
        self.d4 = UNetDown(ngf * 4,     ngf * 8)
        self.d5 = UNetDown(ngf * 8,     ngf * 8)
        self.d6 = UNetDown(ngf * 8,     ngf * 8)
        self.d7 = UNetDown(ngf * 8,     ngf * 8)
        self.d8 = UNetDown(ngf * 8,     ngf * 8, normalize=False)

        self.u1 = UNetUp(ngf * 8,     ngf * 8, dropout=0.5)
        self.u2 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.u3 = UNetUp(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.u4 = UNetUp(ngf * 8 * 2, ngf * 8)
        self.u5 = UNetUp(ngf * 8 * 2, ngf * 4)
        self.u6 = UNetUp(ngf * 4 * 2, ngf * 2)
        self.u7 = UNetUp(ngf * 2 * 2, ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x);  d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8, d7); u2 = self.u2(u1, d6); u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4); u5 = self.u5(u4, d3); u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN. Input: cat(label, image) along channels."""

    def __init__(self, in_channels=3, ndf=64):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride,
                                padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels * 2, ndf,         normalize=False),
            *block(ndf,             ndf * 2),
            *block(ndf * 2,         ndf * 4),
            *block(ndf * 4,         ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, label, img):
        return self.model(torch.cat([label, img], dim=1))


def init_weights(m):
    """Weight init recipe from the Pix2Pix paper: N(0, 0.02)."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def build_baseline_pix2pix(in_channels=3, out_channels=3, ngf=64, ndf=64, device=None):
    """Instantiate G + D, apply init, and (optionally) move to a device."""
    G = UNetGenerator(in_channels=in_channels, out_channels=out_channels, ngf=ngf)
    D = PatchGANDiscriminator(in_channels=in_channels, ndf=ndf)
    G.apply(init_weights)
    D.apply(init_weights)
    if device is not None:
        G.to(device)
        D.to(device)
    return G, D
