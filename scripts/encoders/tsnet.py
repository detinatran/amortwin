"""A temporal-spatial encoder, as a drop-in replacement for cnn_baseline.net().

The U-Net of cnn_baseline treats its 4F input channels as an ordinary image: the very first
3x3 conv mixes every frame of every stream into one 48-channel feature map, so the time axis is
gone before any spatial processing happens. The curve shape -- when contrast arrives, how fast it
washes out -- is exactly what the symbolic decoder needs, and it is the first thing that is thrown
away.

This encoder keeps the time axis explicit for one stage. Each pixel's 4-stream curve is embedded
per frame, attended over time (so arrival and washout are compared frame to frame rather than
summed by a conv), pooled to a vector, and only then handed to the same U-Net trunk as before.

The trunk, its channel widths, normalisation, dropout and head are deliberately identical to
cnn_baseline.net, so a comparison isolates the temporal front end and nothing else.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class TemporalFront(nn.Module):
    """(B, 4F, S, S) -> (B, C, S, S), with the F frames attended over before they are collapsed.

    Every pixel is a token sequence of length F whose elements are the 4 streams at that frame.
    Attention is shared across pixels, so this costs F*F per pixel, not (S*S)^2.
    """

    def __init__(self, F, c_out=48, d=32, heads=4):
        super().__init__()
        self.F, self.d, self.heads = F, d, heads
        self.embed = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.zeros(1, F, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        # mean and max over time: washout slope and peak height matter for different parameters
        self.proj = nn.Conv2d(2 * d, c_out, 1)

    # One pixel is one sequence, so a batch of 16 slices is 262144 sequences. SDPA launches a grid
    # over that dimension and fails with "invalid configuration argument" past ~32k, so the
    # sequences go through in chunks. This is a launch-geometry limit, not a memory one.
    CHUNK = 16384

    def _attn(self, h):
        """Self-attention over the F frames of each pixel.

        nn.MultiheadAttention materialises a (tokens, heads, F, F) score matrix, which is 15 GB for
        one batch here. scaled_dot_product_attention keeps it fused, so memory stays linear in F.
        """
        N, F, d = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (N, F, self.heads, d // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        outs = []
        for i in range(0, N, self.CHUNK):
            s = slice(i, i + self.CHUNK)
            outs.append(torch.nn.functional.scaled_dot_product_attention(q[s], k[s], v[s]))
        o = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
        return self.out(o.transpose(1, 2).reshape(N, F, d))

    def forward(self, x):
        B, C4, S, _ = x.shape
        F = self.F
        # (B, 4F, S, S) -> (B*S*S, F, 4); the 4 streams are stacked blockwise, not interleaved
        t = x.view(B, 4, F, S, S).permute(0, 3, 4, 2, 1).reshape(B * S * S, F, 4)
        h = self.embed(t) + self.pos
        h = h + self._attn(self.norm1(h))
        h = h + self.ff(self.norm2(h))
        h = torch.cat([h.mean(1), h.max(1).values], -1)          # (B*S*S, 2d)
        h = h.view(B, S, S, -1).permute(0, 3, 1, 2)
        return self.proj(h)


class TSNet(nn.Module):
    def __init__(self, F, cin, cout, c=(48, 96, 192, 384), drop=0.2):
        super().__init__()
        self.front = TemporalFront(F, c_out=c[0])
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


def net_ts(F, cin=None, cout=2):
    return TSNet(F, cin or 4 * F, cout)
