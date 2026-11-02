"""Two established segmentation trunks as baselines, so the comparison is not U-Net against
variations of itself.

The table so far compares AmorTwin-S with the CNN, the control and AmorTwin-2D -- all of which
share the cnn_baseline trunk. A reader cannot tell from that whether the trunk is simply weak.
These two replace the trunk with architectures the stroke-imaging literature actually uses:

* AttUNet -- attention gates on the skip connections (Oktay et al. 2018). This is the design
  PerfU-Net adopts for CT perfusion source data, where the gate is meant to suppress the
  irrelevant background that a 4D perfusion stack carries into the decoder.
* ResUNet -- a residual-encoder U-Net. The nnU-Net variants that won ISLES'24 used a large
  residual encoder, so this is the closest stand-in for "what a strong, current pipeline would
  put here" that fits the same input/output contract and the same parameter budget.

Both keep everything else identical: the same 4F input channels, the same dual heads, the same
losses, the same symbolic decoder when AmorTwin-S is trained on top. Only the trunk changes, so a
difference is attributable to the trunk and nothing else. Widths are set to land near the
cnn_baseline parameter count (4.44 M) rather than to maximise capacity -- the question is whether
the trunk is the bottleneck, not whether a bigger network helps.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Pre-activation residual block; the projection handles the channel change."""

    def __init__(self, i, o, drop=0.0):
        super().__init__()
        self.body = block(i, o, drop)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class AttGate(nn.Module):
    """Additive attention gate (Oktay et al.): the decoder signal g selects what the skip x passes."""

    def __init__(self, c_x, c_g, c_int):
        super().__init__()
        self.wx = nn.Conv2d(c_x, c_int, 1)
        self.wg = nn.Conv2d(c_g, c_int, 1)
        self.psi = nn.Sequential(nn.SiLU(), nn.Conv2d(c_int, 1, 1), nn.Sigmoid())

    def forward(self, x, g):
        return x * self.psi(self.wx(x) + self.wg(g))


class _UNetBase(nn.Module):
    def __init__(self, cin, cout, c, drop, blk):
        super().__init__()
        self.enc = nn.ModuleList([blk(cin, c[0]), blk(c[0], c[1]), blk(c[1], c[2])])
        self.mid = blk(c[2], c[3], drop)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([blk(2 * c[2], c[2], drop), blk(2 * c[1], c[1], drop),
                                  blk(2 * c[0], c[0], drop)])
        self.head = nn.Conv2d(c[0], cout, 1)
        self.pool = nn.MaxPool2d(2)
        self.cout = cout
        self.c = c

    def trunk(self, x):
        skips = []
        h = x
        for e in self.enc:
            h = e(h)
            skips.append(h)
            h = self.pool(h)
        h = self.mid(h)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            h = d(torch.cat([u(h), s], 1))
        return h

    def forward(self, x):
        return self.head(self.trunk(x))


class ResUNet(_UNetBase):
    def __init__(self, F, cin, cout, c=(48, 96, 192, 376), drop=0.2):
        super().__init__(cin, cout, c, drop, ResBlock)


class AttUNet(_UNetBase):
    def __init__(self, F, cin, cout, c=(48, 96, 192, 376), drop=0.2):
        super().__init__(cin, cout, c, drop, block)
        self.gates = nn.ModuleList([AttGate(c[2], c[2], c[2] // 2),
                                    AttGate(c[1], c[1], c[1] // 2),
                                    AttGate(c[0], c[0], c[0] // 2)])

    def trunk(self, x):
        skips = []
        h = x
        for e in self.enc:
            h = e(h)
            skips.append(h)
            h = self.pool(h)
        h = self.mid(h)
        for u, d, g, s in zip(self.up, self.dec, self.gates, reversed(skips)):
            hu = u(h)
            h = d(torch.cat([hu, g(s, hu)], 1))     # the gate is driven by the decoder signal
        return h


def net_res(F, cin=None, cout=2):
    return ResUNet(F, cin or 4 * F, cout)


def net_att(F, cin=None, cout=2):
    return AttUNet(F, cin or 4 * F, cout)
