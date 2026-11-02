import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

from cnn_baseline import net


class S4D(nn.Module):
    def __init__(self, d, n=16, dt_min=0.05, dt_max=1.0):
        super().__init__()
        self.d, self.n = d, n
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_a = nn.Parameter(torch.full((d, n), math.log(0.5)))
        self.a_im = nn.Parameter(math.pi * torch.arange(n, dtype=torch.float32).repeat(d, 1))
        self.c = nn.Parameter(torch.randn(d, n, 2) / math.sqrt(n))
        self.skip = nn.Parameter(torch.randn(d))

    def discrete(self):
        a = -torch.exp(self.log_a) + 1j * self.a_im
        da = a * torch.exp(self.log_dt)[:, None]
        bbar = (torch.exp(da) - 1.0) / a
        return da, bbar, torch.view_as_complex(self.c.contiguous())

    def kernel(self, T):
        da, bbar, c = self.discrete()
        t = torch.arange(T, device=da.device, dtype=torch.float32)
        k = torch.einsum("dn,dnt->dt", c * bbar, torch.exp(da[..., None] * t))
        return 2.0 * k.real

    def forward(self, u):
        T = u.shape[-1]
        n = 2 * T
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(self.kernel(T), n=n), n=n)[..., :T]
        return y + self.skip[None, :, None] * u

    def recurrent(self, u):
        da, bbar, c = self.discrete()
        abar = torch.exp(da)
        state = torch.zeros(u.shape[0], self.d, self.n, dtype=abar.dtype, device=u.device)
        out = []
        for t in range(u.shape[-1]):
            state = state * abar + bbar * u[:, :, t, None]
            out.append(2.0 * (state * c).sum(-1).real)
        return torch.stack(out, -1) + self.skip[None, :, None] * u


class SSMBlock(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.ssm = S4D(d, n)
        self.mix = nn.Linear(d, 2 * d)

    def forward(self, x, recurrent=False):
        u = self.norm(x).transpose(1, 2)
        y = self.ssm.recurrent(u) if recurrent else self.ssm(u)
        return x + Fn.glu(self.mix(Fn.gelu(y.transpose(1, 2))), dim=-1)


class SSMFront(nn.Module):
    CHUNK = 8192

    def __init__(self, F, c_out=48, d=32, n=16, depth=2):
        super().__init__()
        self.F, self.d = F, d
        self.embed = nn.Linear(5, d)
        self.blocks = nn.ModuleList([SSMBlock(d, n) for _ in range(depth)])
        self.proj = nn.Conv2d(2 * d, c_out, 1)
        self.register_buffer("clock", torch.arange(F, dtype=torch.float32) / F)

    def lengths(self, x):
        B, _, S, _ = x.shape
        return x.view(B, 4, self.F, S, S)[:, 1, :, 0, 0].sum(1).round().long().clamp(1, self.F)

    def sequences(self, x):
        B, _, S, _ = x.shape
        t = x.view(B, 4, self.F, S, S).permute(0, 3, 4, 2, 1).reshape(B * S * S, self.F, 4)
        return torch.cat([t, self.clock[None, :, None].expand(t.shape[0], -1, 1)], -1)

    def states(self, seq, recurrent=False):
        h = self.embed(seq)
        for b in self.blocks:
            h = b(h, recurrent)
        return h

    def readout(self, h, idx):
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        seen = torch.arange(self.F, device=h.device)[None, :] <= idx[:, None]
        return torch.cat([last, h.masked_fill(~seen[..., None], float("-inf")).amax(1)], -1)

    def chunk(self, part, *idx_part):
        h = self.states(part)
        return tuple(self.readout(h, i) for i in idx_part)

    def encode(self, seq, idxs):
        outs = [[] for _ in idxs]
        for s in range(0, seq.shape[0], self.CHUNK):
            args = (seq[s:s + self.CHUNK],) + tuple(i[s:s + self.CHUNK] for i in idxs)
            if self.training and torch.is_grad_enabled():
                res = checkpoint(self.chunk, *args, use_reentrant=False)
            else:
                res = self.chunk(*args)
            for o, r in zip(outs, res):
                o.append(r)
        return [torch.cat(o, 0) for o in outs]

    def forward_at(self, x, lengths):
        B, _, S, _ = x.shape
        idxs = [(torch.as_tensor(L, device=x.device).long().reshape(-1).expand(B).clamp(1, self.F) - 1)[:, None]
                .expand(B, S * S).reshape(-1) for L in lengths]
        feats = self.encode(self.sequences(x), idxs)
        return [self.proj(f.view(B, S, S, -1).permute(0, 3, 1, 2)) for f in feats]

    def forward(self, x, lengths=None):
        return self.forward_at(x, [self.lengths(x) if lengths is None else lengths])[0]


class SSMNet(nn.Module):
    def __init__(self, F, cin, cout, c0=48):
        super().__init__()
        self.front = SSMFront(F, c0)
        self.trunk = net(F, cin=c0, cout=cout)

    def forward(self, x, lengths=None):
        return self.trunk(self.front(x, lengths))

    def forward_at(self, x, lengths):
        return [self.trunk(f) for f in self.front.forward_at(x, lengths)]


def net_ssm(F, cin=None, cout=2):
    return SSMNet(F, cin or 4 * F, cout)
