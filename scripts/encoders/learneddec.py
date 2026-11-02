"""A learned decoder with the same interface as the symbolic one, to separate two explanations.

AmorTwin-S beats the control by 7.4 points forward and up to 8.5 reverse, and the control differs
from it in TWO ways at once: it has no reconstruction term, and it has no transport model. That
leaves the reader unable to tell whether the gain comes from

  (a) the indicator-dilution model specifically, or
  (b) merely adding a self-supervised reconstruction objective of any kind.

This decoder isolates (b). It consumes exactly what the symbolic decoder consumes -- the encoder's
three per-pixel quantities, the patient's arterial input, and the cut length -- and emits a curve
of the same shape, so the training loop, the loss and the readout are untouched. What it does not
have is the model: no convolution with a residue function, no delay-shifted AIF, no amplitude or
transit scale with a physical meaning. It is an MLP that may learn any mapping it likes.

Capacity is set ABOVE the symbolic decoder's, not below: the symbolic one has zero weights, so any
positive number of parameters is already more, and a small MLP (two hidden layers, 64 units,
8.1 k parameters) is given the AIF as a 16-dimensional learned embedding rather than the raw 60
frames so that it is a fair function approximator rather than a lookup table. If the gain were
about reconstruction per se, this arm should recover most of it.
"""
import torch
import torch.nn as nn


class LearnedDecoder(nn.Module):
    """(theta (P,3), aif (F,), L) -> rebuilt curve (P, L), with no transport model inside."""

    def __init__(self, F, d_aif=16, hidden=64):
        super().__init__()
        self.F = F
        # the AIF enters as a learned summary, the same information the symbolic decoder gets
        self.aif_embed = nn.Sequential(nn.Linear(F, 32), nn.SiLU(), nn.Linear(32, d_aif))
        # per-pixel: theta (3) + aif embedding (d_aif) + a time code -> curve value at that time
        self.net = nn.Sequential(
            nn.Linear(3 + d_aif + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    CHUNK = 4096

    def forward(self, theta, aif, L):
        """Chunked over pixels: the concatenated (P, L, 3+d_aif+2) input and the activations of a
        two-layer MLP over it are the memory cost here, and at P ~ 16k pixels and L = 60 frames a
        single slice reached 5.7 GB -- four times the symbolic decoder, which stores nothing at
        all. Chunking caps the live tensor without changing the result."""
        a = self.aif_embed(aif[None, :])                      # (1, d_aif)
        t = torch.arange(L, device=theta.device, dtype=theta.dtype) / max(self.F, 1)
        # time code: linear ramp plus the AIF value at that frame, so the decoder can be causal
        tc = torch.stack([t, aif[:L]], -1)                    # (L, 2)
        outs = []
        for s in range(0, theta.shape[0], self.CHUNK):
            th_c = theta[s:s + self.CHUNK]
            n = th_c.shape[0]
            th = th_c[:, None, :].expand(n, L, 3)
            ae = a[:, None, :].expand(n, L, a.shape[-1])
            tt = tc[None].expand(n, L, 2)
            outs.append(self.net(torch.cat([th, ae, tt], -1))[..., 0])
        return torch.cat(outs, 0)                             # (P, L)


def learned_recon_loss(dec, theta, curve, aif, L, logA):
    """Floored nMSE on the observed frames, mirroring at2d_tta.recon_loss so the two arms are
    scored identically; the amplitude stays a free per-patient scalar, as there too."""
    y = dec(theta, aif, L)
    y = torch.exp(logA) * y
    num = ((y - curve) ** 2).mean(1)
    den = (curve ** 2).mean(1).clamp_min(0.05 * (curve ** 2).mean().detach().clamp_min(1e-3))
    return (num / den).clamp(max=10.0).mean()
