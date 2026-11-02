"""FineNet: a full-resolution fine path beside the trunk, for the lesions the pooling erases.

THE GAP IN THE NINE ARMS TRIED

Grouped by where they intervene, against the plain U-Net encoder of AmorTwin-S:

    encoder, temporal (TSNet, CBNet, DuoNet-2)   |core| 6.1-8.7 ml   agreement in seed noise
    encoder, head (DuoNet, DuoNet-3, FloorNet)   |core| 7.1-8.4      agreement in seed noise
    readout / decision loss (CalNet, GapNet,     |core| 10.7-19.8    agreement worse
                             MarginNet)

Every arm that changed the encoder changed its TEMPORAL view. None changed its SPATIAL resolution,
and the measurements point there:

  * the oracle study shows CORE is the binding quantity (substituting the true core raises
    agreement to 95.4-97.2 %; substituting the true hypo LOWERS it), so core precision is the one
    volume worth buying;
  * relative core error is ~1.0 on sub-5 ml lesions and 0.3-0.4 above 20 ml, and 24 of 30 fold-3
    patients sit below 20 ml;
  * on small lesions the mask head places only ~30 % of the reference core's mass inside the
    reference lesion (2.5 of 8.4 ml), against ~42 % for large ones -- the signature of spatial
    dilution, not of a missing curve feature;
  * the trunk pools 128 -> 16: an 8 ml lesion is 1-2 units wide at the bottleneck, and the skip
    connections pass through 3x3 convs that have already mixed it with background.

THE CHANGE, AND ONLY THIS CHANGE

A thin, never-pooled path at full resolution -- three 3x3 convs with dilations 1, 2, 4 (receptive
field ~15 px, ~0.1 M parameters) -- concatenated with the decoder's final feature map before the
head. The trunk, the losses and the readout stay exactly cnn_baseline's; no temperature, no extra
loss term, so a comparison against the U-Net isolates the fine path. This stays inside the one
intervention class that never damaged the volumes (10/10 record: every readout/loss intervention
made |core| worse; every encoder intervention kept it at or below the U-Net).

PRE-DECLARED EXPECTATION AND READOUT (written before training)

Primary: |core| at 30 s on fold 3 against the U-Net's 8.6 ml (seeds 7.8-10.1), and the fraction of
core mass inside the reference lesion for sub-20 ml patients (baseline ~30 %). Agreement is
expected to stay inside seed noise: the oracle counterfactual caps every accuracy-seeking
architecture, and the error correlation across arms is r = 0.76-0.82. If |core| does not improve,
the dilution hypothesis is wrong and this line stops.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class FinePath(nn.Module):
    """Full-resolution, small, dilated: detail the pooled trunk cannot keep."""

    def __init__(self, cin, c=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=2, dilation=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=4, dilation=4), nn.GroupNorm(8, c), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class FineNet(nn.Module):
    def __init__(self, F, cin, cout, c=(48, 96, 192, 384), drop=0.2, cf=32):
        super().__init__()
        self.enc = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3], drop)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(2 * c[2], c[2], drop), block(2 * c[1], c[1], drop),
                                  block(2 * c[0], c[0], drop)])
        self.fine = FinePath(cin, cf)
        self.merge = nn.Conv2d(c[0] + cf, c[0], 1)
        self.head = nn.Conv2d(c[0], cout, 1)
        self.pool = nn.MaxPool2d(2)
        self.cout = cout

    def forward(self, x):
        skips = []
        h = x
        for e in self.enc:
            h = e(h)
            skips.append(h)
            h = self.pool(h)
        h = self.mid(h)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            h = d(torch.cat([u(h), s], 1))
        h = self.merge(torch.cat([h, self.fine(x)], 1))
        return self.head(h)


def net_fine(F, cin=None, cout=2):
    return FineNet(F, cin or 4 * F, cout)
