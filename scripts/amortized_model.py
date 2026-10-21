"""AmorTwin: the amortized physics twin network (shared by training, inference and evaluation).

Per voxel, the input is (6, TMAX): the voxel curve and the patient's median tissue curve, both
divided by the voxel's / median curve's maximum, the normalised AIF, the measured-frame mask,
the time, and the broadcast log ratio of the two maxima (amplitude relative to the brain).
Tissue class (grey/white matter) enters as two extra constant channels. The output is the
three twin voxel targets: ln(MTT / brain median), delay (s), ln(CBV / brain median).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TMAX = 70
IN_CH = 8
IN_CH_ROI = 10                             # + the median curve of the voxel's ROI and its log amplitude ratio
TARGETS = ("ln_rel_MTT", "delay_s", "ln_rel_CBV")
TARGETS4 = TARGETS + ("tmax_s",)          # truncation-aware model (E3): + the twin Tmax
TARGETS6 = TARGETS4 + ("ln_MTT_s", "ln_beta_s")   # + ln MTT and ln ROI dispersion: Tmax composed by physics


def residue_peak_t(beta, mtt):
    """Torch: time to peak of exp(-t/MTT) through the dispersion kernel exp(-t/beta)/beta."""
    beta, mtt = beta.clamp(min=1e-3), mtt.clamp(min=1e-3)
    close = (mtt - beta).abs() < 1e-3 * beta
    t = beta * mtt * torch.log(mtt / beta) / torch.where(close, torch.ones_like(mtt), mtt - beta)
    return torch.where(close, beta, t)


def composed_tmax(mean, sd, n=64, gen=None):
    """Tmax = delay + residue peak(beta, MTT) from the TARGETS6 Gaussian posterior (independent
    delay, ln MTT, ln beta), by Monte Carlo: returns its mean and SD (torch, per row)."""
    e = torch.randn((n,) + mean.shape[:1] + (3,), device=mean.device, generator=gen)
    j = [1, 4, 5]
    x = mean[None, :, j] + e * sd[None, :, j]
    tm = x[..., 0] + residue_peak_t(torch.exp(x[..., 2]), torch.exp(x[..., 1]))
    return tm.mean(0), tm.std(0)


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        return x + self.c2(F.gelu(self.n(self.c1(F.gelu(x)))))


class AmorTwin(nn.Module):
    def __init__(self, c=64, n_out=len(TARGETS), in_ch=IN_CH):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16, 1, 2, 4)])
        self.head = nn.Sequential(nn.Linear(2 * c, 128), nn.GELU(), nn.Linear(128, n_out))

    def forward(self, x):
        h = self.blocks(self.inp(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))




def make_inputs(curve, ctx, aif, length, tissue, roi=None):
    """numpy/torch-agnostic builder: curve (N, TMAX), ctx (N, TMAX), aif (N, TMAX), length (N,),
    tissue (N,) -> float32 tensor (N, IN_CH, TMAX); with roi (N, TMAX), the median curve of the
    voxel's ROI, two more channels (that curve over its maximum, and the log ratio of its maximum to
    the brain median's) -> (N, IN_CH_ROI, TMAX)."""
    curve = torch.as_tensor(curve, dtype=torch.float32)
    ctx = torch.as_tensor(ctx, dtype=torch.float32)
    aif = torch.as_tensor(aif, dtype=torch.float32)
    length = torch.as_tensor(length)
    tissue = torch.as_tensor(tissue)
    N = curve.shape[0]
    t = torch.arange(TMAX, device=curve.device)[None, :].expand(N, -1)
    mask = (t < length[:, None]).float()
    s = curve.amax(1).clamp(min=1e-3)
    sc = ctx.amax(1).clamp(min=1e-3)
    ratio = torch.log(s / sc)[:, None].expand(-1, TMAX)
    gm = (tissue == 1).float()[:, None].expand(-1, TMAX)
    wm = (tissue == 2).float()[:, None].expand(-1, TMAX)
    ch = [curve / s[:, None] * mask, ctx / sc[:, None] * mask, aif * mask, mask, t.float() / TMAX, ratio, gm, wm]
    if roi is not None:
        roi = torch.as_tensor(roi, dtype=torch.float32)
        sr = roi.amax(1).clamp(min=1e-3)
        ch += [roi / sr[:, None] * mask, torch.log(sr / sc)[:, None].expand(-1, TMAX)]
    return torch.stack(ch, 1)


def predict(model, curve, ctx, aif, length, tissue, batch=65536, device="cpu"):
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(curve), batch):
            x = make_inputs(curve[s:s + batch], ctx[s:s + batch], aif[s:s + batch], length[s:s + batch],
                            tissue[s:s + batch]).to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)
