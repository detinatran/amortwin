"""A curve-basis encoder: a cheap temporal front end for the symbolic decoder.

tsnet.py gives every pixel its own attention sequence, which costs 8.4 GB at a batch of 4 and caps
the 3080 Ti there. That cost buys frame-to-frame comparison the decoder never asks for: the decoder
wants three numbers per pixel (ln MTT, delay, ln CBV), and an indicator-dilution curve is almost
entirely described by when contrast arrives, how high it peaks and how fast it washes out.

Such curves live in a low-dimensional space, so instead of attending over 60 frames this projects
each pixel's 4 streams onto K learned temporal bases -- one matmul, no per-pixel sequence. The
bases are initialised as a gamma-variate family spanning plausible arrival times and widths, which
is where perfusion curves actually sit, and are then free to move.

The trunk, its channel widths, normalisation, dropout and head are identical to cnn_baseline.net,
so a comparison isolates the temporal front end and nothing else.
"""
import math

import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


def gamma_basis(F, K):
    """K gamma-variate curves over F frames, spread over arrival time and width.

    g(t) = ((t - t0) / a)^alpha exp(-(t - t0) / a), the standard shape of a contrast bolus. Spanning
    t0 and a covers early/late arrival and sharp/dispersed transit, i.e. exactly the axes the
    decoder's parameters move along.
    """
    t = torch.arange(F, dtype=torch.float32)
    n_t0 = max(2, K // 2)                 # arrival times per width group
    n_a = max(1, (K + n_t0 - 1) // n_t0)  # width groups
    out = []
    for i in range(K):
        t0 = 2.0 + 10.0 * (i % n_t0) / (n_t0 - 1)
        a = 2.0 + 4.0 * min(i // n_t0, n_a - 1)
        u = ((t - t0) / a).clamp(min=0.0)
        g = u.pow(2.0) * torch.exp(-u)
        n = g.norm()
        out.append(g / n if float(n) > 1e-6 else g)
    return torch.stack(out, 0)                                   # (K, F)


class CurveBasisFront(nn.Module):
    """(B, 4F, S, S) -> (B, c_out, S, S).

    Each of the 4 streams is projected onto K temporal bases, giving 4K coefficients per pixel.
    That is a single einsum over the frame axis: no per-pixel sequence, no attention matrix.
    """

    def __init__(self, F, c_out=48, K=12):
        super().__init__()
        self.F, self.K = F, K
        self.basis = nn.Parameter(gamma_basis(F, K))             # learned, gamma-variate at init
        self.scale = nn.Parameter(torch.ones(1))
        self.mix = nn.Sequential(
            nn.Conv2d(4 * K, 2 * c_out, 1), nn.GroupNorm(8, 2 * c_out), nn.SiLU(),
            nn.Conv2d(2 * c_out, c_out, 1),
        )

    def forward(self, x):
        B, _, S, _ = x.shape
        F, K = self.F, self.K
        v = x.view(B, 4, F, S, S)
        # (B, 4, F, S, S) x (K, F) -> (B, 4, K, S, S); normalising by F keeps the scale of the
        # coefficients independent of how many frames the cut left.
        c = torch.einsum("bcfij,kf->bckij", v, self.basis) * (self.scale / math.sqrt(F))
        return self.mix(c.reshape(B, 4 * K, S, S))


class CBNet(nn.Module):
    def __init__(self, F, cin, cout, c=(48, 96, 192, 384), drop=0.2, K=12):
        super().__init__()
        self.front = CurveBasisFront(F, c_out=c[0], K=K)
        self.enc = nn.ModuleList([block(c[0], c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3], drop)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(2 * c[2], c[2], drop), block(2 * c[1], c[1], drop),
                                  block(2 * c[0], c[0], drop)])
        self.head = nn.Conv2d(c[0], cout, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.front(x)
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], 1))
        return self.head(x)


def net_cb(F, cin=None, cout=2, K=12):
    return CBNet(F, cin or 4 * F, cout, K=K)
