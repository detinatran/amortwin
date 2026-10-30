"""CalNet: the readout is a volume, so train the network on the volume.

Measured on the held-out fold (U-Net, 30 s cut), two things are true at once:

    mean |core error|  7.8 ml        mean |hypo error|  37.1 ml
    hypo bias         +19.9 ml, and it does NOT shrink on the full scan (+22.4)

so the hypoperfused volume is wrong by a systematic, truncation-independent amount. Looking at the
probabilities explains it: **59.5 % of hypoperfused voxels sit in (0.2, 0.8)**, and only 18.5 %
exceed 0.8. The network is undecided almost everywhere, and since the readout is

    V = sum_v p(v) w(v)

every one of those undecided voxels contributes a fraction of a millilitre. The volume is set by the
uncommitted mass, not by the confident core. A per-voxel cross-entropy never sees this: it is
summed over voxels, so it is indifferent to whether the errors cancel in the sum that is actually
read out. Ratios confirm the asymmetry: small lesions come out x1.6-x2.3, large ones x0.8.

CalNet keeps the U-Net trunk and changes what the mask head is asked for:

  1. a differentiable volume loss. V is literally sum p*w, so it is differentiable in p: the
     network is trained on the quantity DEFUSE 3 consumes, not only on per-voxel labels.
  2. a learned temperature per class. One scalar, applied to the logits, lets the network sharpen
     or soften its probabilities globally -- the cheapest correct fix for miscalibration, and the
     standard one (Guo et al., temperature scaling), except that here it is fitted jointly rather
     than post hoc, because the volume loss needs it during training.
  3. a decisiveness penalty that pushes mass out of (0.2, 0.8), applied only where the two classes
     agree, so genuinely ambiguous penumbra is left alone.

What it deliberately does NOT do is threshold. The paper shows a thresholded readout costs 30-62
points; the aim here is a *calibrated* soft readout, which is a different thing: thresholding
discards the fractional mass, calibration makes it correct.

The trunk, widths, normalisation and dropout are identical to cnn_baseline.net, so a comparison
isolates the head and the loss.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class CalNet(nn.Module):
    """U-Net trunk; the mask logits pass through a learned per-class temperature."""

    def __init__(self, F, cin, cout, c=(48, 96, 192, 384), drop=0.2):
        super().__init__()
        self.enc = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3], drop)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(2 * c[2], c[2], drop), block(2 * c[1], c[1], drop),
                                  block(2 * c[0], c[0], drop)])
        self.head = nn.Conv2d(c[0], cout, 1)
        self.pool = nn.MaxPool2d(2)
        self.cout = cout
        # one temperature per mask class (core, hypo); log-parameterised so it stays positive
        self.log_temp = nn.Parameter(torch.zeros(2))

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
        out = self.head(h)
        if self.cout >= 10:
            # scale only the mask logits; the physical channels are untouched
            t = torch.exp(self.log_temp).clamp(0.2, 5.0)[None, :, None, None]
            out = torch.cat([out[:, :8], out[:, 8:10] / t], 1)
        return out


def net_cal(F, cin=None, cout=2):
    return CalNet(F, cin or 4 * F, cout)


def volume_loss(logits, target, valid, pixel_ml=1.0):
    """Relative error of the volume the readout actually computes, V = sum p*w.

    Per-voxel cross-entropy is indifferent to how errors combine in that sum; this term is not.
    It is relative, so one large lesion does not dominate a batch of small ones, and it is applied
    per slice, which is the unit the network sees.
    """
    p = torch.sigmoid(logits)
    v = valid.float()
    pv = (p * v).flatten(2).sum(-1) * pixel_ml          # (B, 2) predicted
    tv = (target * v).flatten(2).sum(-1) * pixel_ml     # (B, 2) reference
    return ((pv - tv).abs() / tv.clamp_min(1.0)).mean()


def decisiveness(logits, valid, lo=0.2, hi=0.8):
    """Push probability mass out of the undecided band, where the two classes agree.

    p(1-p) is maximal at 0.5 and vanishes at either end. Masking to voxels where core and hypo
    point the same way leaves the genuinely ambiguous penumbra -- the boundary between them -- free
    to stay uncertain, which is where uncertainty is real rather than a failure to commit.
    """
    p = torch.sigmoid(logits)
    agree = ((p[:, 0] > 0.5) == (p[:, 1] > 0.5)).float()[:, None]
    band = ((p > lo) & (p < hi)).float()
    return (p * (1 - p) * band * agree * valid.float()).mean()
