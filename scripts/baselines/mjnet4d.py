"""4D mJ-Net, ported from 4d-mJ-Net (Architectures/arch_mJNet.py:mJNet_4D).

The reference runs on TensorFlow 2.1 / Keras 2.3, which needs CUDA 10.1 and so cannot run on
any card here (sm_86, sm_89). The parts that matter port directly: Conv4D is a loop of
weight-shared 3D convolutions (see conv4d.py, verified against the TF original to 9.5e-07),
and the rest is an ordinary 3D U-Net with additive attention gates.

Shape convention follows our own data rather than theirs: input is (b, 1, z, t, h, w) built
straight from the `ctc` array, with no pickle pipeline in between.

Structure, as in the reference:
  three Conv4D blocks, each ending in a strided Conv4D that halves the time axis
  reshape to 3D, one Conv3d block that collapses z
  a 2D U-Net decoder with attention gates on the skips
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv4d import Conv4D


class Block4D(nn.Module):
    """block_conv4D: (len(channels)-1) plain Conv4D, then one strided Conv4D."""

    def __init__(self, in_ch, channels, kernel_size, reduce_dim=2, stride_t=2, batch=True):
        super().__init__()
        layers, c = [], in_ch
        for ch in channels[:-1]:
            layers += [Conv4D(c, ch, kernel_size, reduce_dim=reduce_dim),
                       nn.LeakyReLU(0.33)]
            if batch:
                layers.append(nn.BatchNorm3d(ch))   # applied after the reshape in forward
            c = ch
        self.body = nn.ModuleList(layers)
        self.last = Conv4D(c, channels[-1], kernel_size, reduce_dim=reduce_dim)
        self.act = nn.LeakyReLU(0.33)
        self.bn = nn.BatchNorm3d(channels[-1]) if batch else nn.Identity()
        self.stride_t = stride_t

    def _bn6(self, bn, x):
        """BatchNorm3d over a 6D tensor: fold z and t together, then restore."""
        b, c, z, t, h, w = x.shape
        return bn(x.reshape(b, c, z * t, h, w)).reshape(b, c, z, t, h, w)

    def forward(self, x):
        for m in self.body:
            x = self._bn6(m, x) if isinstance(m, nn.BatchNorm3d) else m(x)
        x = self.act(self.last(x))
        x = self._bn6(self.bn, x) if isinstance(self.bn, nn.BatchNorm3d) else x
        # the reference strides inside the last Conv4D; striding the time axis afterwards is
        # equivalent for stride on that axis alone and keeps Conv4D's 'same' padding simple
        return x[:, :, :, ::self.stride_t]


class AttentionGate(nn.Module):
    """block_attentionGate: additive attention on a skip connection."""

    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        self.theta = nn.Conv2d(x_ch, inter_ch, 2, stride=2, bias=False)
        self.phi = nn.Conv2d(g_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)
        self.out = nn.Sequential(nn.Conv2d(x_ch, x_ch, 1), nn.BatchNorm2d(x_ch))

    def forward(self, x, g):
        t = self.theta(x)
        p = self.phi(g)
        if p.shape[-2:] != t.shape[-2:]:
            p = F.interpolate(p, size=t.shape[-2:], mode="bilinear", align_corners=False)
        a = torch.sigmoid(self.psi(F.relu(t + p)))
        a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(x * a)


def double_conv2d(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.33), nn.BatchNorm2d(o),
                         nn.Conv2d(o, o, 3, padding=1), nn.LeakyReLU(0.33), nn.BatchNorm2d(o))


class MJNet4D(nn.Module):
    """input (b, 1, z, t, h, w) -> (b, out_ch, h, w)."""

    def __init__(self, z=8, t=30, out_ch=2, c=4, limchan=2):
        super().__init__()
        k = (z, 3, 3, 3)   # (z, t, x, y) as the reference builds it for time-first layout
        ch = [max(1, 8 // limchan)] * 3
        self.b1 = Block4D(1, ch, k, reduce_dim=2, stride_t=2)
        self.b2 = Block4D(ch[-1], ch, k, reduce_dim=2, stride_t=2)
        self.b3 = Block4D(ch[-1], ch, k, reduce_dim=2, stride_t=2)
        # collapse z with one Conv3d whose kernel spans it, as block_conv does
        # the reference fixes this kernel to n_slices; ISLES'18 has z = 8 and ISLES'24 z = 16,
        # so a z-sized kernel would make the weights cohort-specific and block transfer.
        # A 3-deep kernel plus the mean below collapses z without tying the net to one cohort.
        self.collapse = nn.Sequential(
            nn.Conv3d(ch[-1], 32 // limchan, (3, 3, 3), padding=(1, 1, 1)),
            nn.LeakyReLU(0.33), nn.BatchNorm3d(32 // limchan))
        self.enc1 = double_conv2d(32 // limchan, 16 * c)
        self.enc2 = double_conv2d(16 * c, 32 * c)
        self.mid = double_conv2d(32 * c, 64 * c)
        self.att2 = AttentionGate(32 * c, 64 * c, 32 * c)
        self.up2 = nn.ConvTranspose2d(64 * c, 32 * c, 2, 2)
        self.dec2 = double_conv2d(64 * c, 32 * c)
        self.att1 = AttentionGate(16 * c, 32 * c, 16 * c)
        self.up1 = nn.ConvTranspose2d(32 * c, 16 * c, 2, 2)
        self.dec1 = double_conv2d(32 * c, 16 * c)
        self.head = nn.Conv2d(16 * c, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))          # (b, ch, z, t', h, w)
        x = x.mean(dim=3)                          # time is spent; average what remains
        x = self.collapse(x)                       # (b, c, z', h, w), z' == 1 when z matches
        x = x.mean(dim=2)                          # collapse whatever z remains -> (b, c, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), self.att2(e2, m)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(e1, d2)], 1))
        return self.head(d1)
