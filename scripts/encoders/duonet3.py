"""DuoNet-3: the identifiability idea, with the three faults that broke DuoNet fixed.

Probing the trained DuoNet checkpoint showed its mechanism never engaged:

    spread at L=25/30/35/full:  2.893 / 2.915 / 2.932 / 2.923   (changes by 1 %)
    median spread:              4.4817 = exp(1.5), the clamp ceiling
    one-sided penalty there:    0.20, versus 2.13 for a narrow spread

So the optimiser did the cheapest thing available: open the spread to its ceiling everywhere and
leave it there. Three faults, each fixable:

1. **Widening was free.** `softplus(target - logs)` only charges for collapsing, so there was no
   force pulling a well-determined pixel back to a point estimate. Fixed by charging for the
   spread itself as well -- a weak quadratic that is negligible at a narrow spread and real at a
   wide one. Collapse is still penalised, so the two terms meet where the data decide.

2. **The cut length never reached the head.** It was passed as `cut_frac` but the training loop
   never supplied it, so the default (all ones) was used on every batch: the head could not
   condition on L even in principle. Fixed by deriving it from the frame mask, which the input
   already carries, instead of relying on the caller.

3. **The spread only perturbed the reconstruction.** Volumes come from the mask head, a separate
   branch, so the noise reached the decoder but never the quantity that is read out. Fixed by
   using the spread where it is actually informative: a pixel whose MTT is undetermined is a pixel
   whose hypoperfusion label is uncertain, so the spread down-weights that pixel's mask loss. That
   is the identifiability result acting on the readout rather than beside it.

The trunk is the U-Net of cnn_baseline, unchanged.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class DuoNet3(nn.Module):
    def __init__(self, F, cin, cout, c=(48, 96, 192, 384), drop=0.2):
        super().__init__()
        self.F = F
        self.enc = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3], drop)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(2 * c[2], c[2], drop), block(2 * c[1], c[1], drop),
                                  block(2 * c[0], c[0], drop)])
        self.head = nn.Conv2d(c[0], cout, 1)
        self.spread = nn.Sequential(nn.Conv2d(c[0] + 1, c[0], 1), nn.SiLU(), nn.Conv2d(c[0], 1, 1))
        self.pool = nn.MaxPool2d(2)
        self.cout = cout
        self.log_temp = nn.Parameter(torch.zeros(2))
        self.last_logs = None

    def forward(self, x):
        # The frame mask occupies channels F..2F of the input, so the observed fraction of the scan
        # is already here. Deriving it means the head sees L whatever the caller does.
        F = self.F
        cut_frac = x[:, F:2 * F].mean(dim=1, keepdim=True)
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
        cf = torch.nn.functional.interpolate(cut_frac, size=h.shape[-2:], mode="nearest")
        self.last_logs = self.spread(torch.cat([h, cf], 1)).clamp(-6.0, 1.5)
        if self.cout >= 10:
            t = torch.exp(self.log_temp).clamp(0.2, 5.0)[None, :, None, None]
            out = torch.cat([out[:, :8], out[:, 8:10] / t], 1)
        return out


def net_duo3(F, cin=None, cout=2):
    return DuoNet3(F, cin or 4 * F, cout)


def sample_mtt(q, logs, training=True):
    """Replace the point ln MTT in q (P, 3) by a sample; delay and ln CBV stay point estimates."""
    if not training:
        return q
    eps = torch.randn_like(q[:, 0])
    return torch.stack([q[:, 0] + eps * torch.exp(logs), q[:, 1], q[:, 2]], 1)


def spread_penalty(logs, target=0.0, wide=0.05):
    """Two-sided, but asymmetric: collapsing costs a lot, widening costs a little.

    DuoNet charged only for collapse, so the spread sat at its ceiling for every pixel and every
    cut. The quadratic term is what stops that: at logs = -2 it adds 0.0002 and changes nothing,
    at the ceiling it adds 0.11 and competes with the 0.20 the collapse term saves. The spread then
    settles where the reconstruction actually prefers it.
    """
    collapse = torch.nn.functional.softplus(target - logs)
    return (collapse + wide * logs.clamp_min(0.0) ** 2).mean()


def mask_weight(logs, floor=0.3):
    """Down-weight the mask loss of pixels whose MTT the data do not determine.

    This is the piece DuoNet lacked. A wide spread means the curve is consistent with many transit
    times, which is exactly the regime where the vendor's Tmax threshold -- and hence the
    hypoperfusion label -- is least trustworthy. Weighting by it lets the identifiability estimate
    reach the readout instead of only the reconstruction. The floor keeps every pixel contributing.
    """
    w = torch.exp(-logs.clamp(-6.0, 1.5))
    return (floor + (1 - floor) * w / w.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)).detach()
