import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.func import jvp

from at2d_tta import SMOOTH, rebuild
from cnn_baseline import net


class FisherRefiner(nn.Module):
    def __init__(self, iters=3):
        super().__init__()
        self.iters = iters
        self.log_m0 = nn.Parameter(torch.tensor(math.log(6.0)))
        self.log_noise = nn.Parameter(torch.tensor(math.log(1e-3)))
        self.prior_raw = nn.Parameter(torch.zeros(3))
        self.step_raw = nn.Parameter(torch.full((iters,), 0.5413))
        self.damp_raw = nn.Parameter(torch.full((iters,), -5.81))
        self.register_buffer("lo", torch.tensor([-3.0, -8.0, -3.0]))
        self.register_buffer("hi", torch.tensor([3.0, 30.0, 3.0]))
        self.register_buffer("trust", torch.tensor([0.7, 3.0, 0.7]))

    def bound(self, th):
        return torch.minimum(torch.maximum(th, self.lo), self.hi)

    def jacobian(self, aif, th, L, m0):
        th = th.detach()
        cols = []
        for i in range(3):
            tan = torch.zeros_like(th)
            tan[:, i] = 1.0
            cols.append(jvp(lambda q: rebuild(aif, q, L, m0)[0], (th,), (tan,))[1])
        return torch.stack(cols, -1)

    def system(self, aif, th, a, c, L, m0, scale, prec0, damp, theta0):
        y, ok = rebuild(aif, th, L, m0)
        okf = ok.to(y.dtype)
        amp = torch.exp(a)
        m = amp * y
        r = okf * (m - c) * scale
        J = amp.detach() * self.jacobian(aif, th, L, m0) * (okf * scale)[..., None]
        u = (m * okf * scale).detach()
        s2 = torch.exp(self.log_noise)
        Hd = torch.einsum("plj,plk->pjk", J, J) / s2
        Hpp = Hd + torch.diag_embed(prec0 + damp * torch.diagonal(Hd, dim1=-2, dim2=-1) + 1e-6)
        Hpa = torch.einsum("plj,pl->pj", J, u) / s2
        Haa = (u * u).sum() / s2 * (1.0 + damp) + 1e-6
        bp = torch.einsum("plj,pl->pj", J, r) / s2 + prec0 * (th - theta0)
        ba = (u * r).sum() / s2
        dt = y.dtype
        Hb = torch.linalg.solve(Hpp.double(), bp.double().unsqueeze(-1)).squeeze(-1).to(dt)
        Ha = torch.linalg.solve(Hpp.double(), Hpa.double().unsqueeze(-1)).squeeze(-1).to(dt)
        S = (Haa - (Hpa * Ha).sum()).clamp(min=1e-8)
        da = (ba - (Hpa * Hb).sum()) / S
        dth = Hb - Ha * da
        return dth, da, Hpp, Ha, S

    def forward(self, theta0, var0, c, aif, L, m0=None):
        # m0 comes from the caller when the training loop owns a per-patient transit scale.
        # Using the refiner's own global log_m0 while at2d_sae rebuilds with logM[patient] put the
        # two on different transit scales: the refined theta no longer explained the curve that
        # recon_loss scored, and training diverged (loss 6.29 -> 11.37 in two epochs).
        m0 = torch.exp(self.log_m0.clamp(math.log(1.5), math.log(30.0))) if m0 is None else m0
        e = (c ** 2).sum(1)
        floor = (0.05 * e.mean().detach()).clamp(min=1e-3)
        scale = torch.rsqrt(torch.maximum(e, floor))[:, None]
        prec0 = torch.exp(self.prior_raw)[None] / var0.clamp(min=1e-6)
        th = theta0
        y, ok = rebuild(aif, self.bound(th), L, m0)
        okf = ok.to(y.dtype)
        a = torch.log(((okf * y * c).sum() / (okf * y * y).sum().clamp_min(1e-12)).clamp(min=1e-4))
        for k in range(self.iters):
            damp = Fn.softplus(self.damp_raw[k])
            dth, da, _, _, _ = self.system(aif, self.bound(th), a, c, L, m0, scale, prec0, damp, theta0)
            step = Fn.softplus(self.step_raw[k])
            th = self.bound(th - torch.maximum(torch.minimum(step * dth, self.trust), -self.trust))
            a = a - (step * da).clamp(-0.7, 0.7)
        _, _, Hpp, Ha, S = self.system(aif, th, a, c, L, m0, scale, prec0, Fn.softplus(self.damp_raw[-1]), theta0)
        diag = torch.diagonal(torch.linalg.inv(Hpp.double()), dim1=-2, dim2=-1).to(Hpp.dtype)
        var = (diag + Ha ** 2 / S).clamp(min=1e-8)
        return th, var


class FisherNet(nn.Module):
    def __init__(self, F, cin, cout, inner=None, iters=3):
        super().__init__()
        self.F, self.cout = F, cout
        self.inner = inner if inner is not None else net(F, cin=cin, cout=cout)
        self.refiner = FisherRefiner(iters)
        self.register_buffer("mu", torch.zeros(4))
        self.register_buffer("sd", torch.ones(4))
        r = int(4 * SMOOTH + 0.5)
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (t / SMOOTH) ** 2)
        self.register_buffer("gk", (k / k.sum())[None, None, :])
        self.pad = r

    def set_norm(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def smooth(self, c):
        return Fn.conv1d(Fn.pad(c[:, None, :], (self.pad, self.pad), mode="replicate"), self.gk)[:, 0, :]

    def refine_slice(self, o, xj, m0=None):
        F = self.F
        S = o.shape[-1]
        L = int(min(max(int(xj[1, :, 0, 0].sum().round()), 3), F))
        c = xj[0, :L].reshape(L, S * S).t()
        energy = (c ** 2).sum(1)
        active = energy > 1e-6 * energy.max().clamp(min=1e-12)
        if int(active.sum()) < 50:
            return o
        flat = o.reshape(o.shape[0], S * S).t().clone()
        th0 = flat[active, :3] * self.sd[:3] + self.mu[:3]
        var0 = torch.exp(flat[active, 4:7].clamp(-8, 6)) * self.sd[:3] ** 2
        th, var = self.refiner(th0, var0, self.smooth(c[active]), xj[2, :, 0, 0], L, m0)
        flat[active, :3] = (th - self.mu[:3]) / self.sd[:3]
        flat[active, 4:7] = torch.log(var) - 2.0 * torch.log(self.sd[:3])
        return flat.t().reshape(o.shape)

    def forward(self, x, m0=None):
        """m0: (B,) transit scale per slice, supplied by the training loop as exp(logM[patient])."""
        out = self.inner(x)
        B, _, S, _ = x.shape
        xv = x.reshape(B, 4, self.F, S, S)
        return torch.stack([self.refine_slice(out[j], xv[j],
                                              None if m0 is None else m0[j])
                            for j in range(B)])


def net_fisher(F, cin=None, cout=2):
    return FisherNet(F, cin or 4 * F, cout)


def net_ssmfisher(F, cin=None, cout=2):
    from ssmnet import SSMNet
    return FisherNet(F, cin or 4 * F, cout, inner=SSMNet(F, cin or 4 * F, cout))
