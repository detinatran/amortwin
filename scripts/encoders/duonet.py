"""DuoNet: two kinds of parameter, handled two ways.

Sec. III-B of the paper proves something about this problem that the encoder never uses: a scan cut
at 25-35 s does not determine the transit time (Cramer-Rao x8.7 at 30 s in hypoperfused tissue).
The delay and the amplitude are well determined by the same curve; MTT is not.

Every encoder here so far -- the U-Net, tsnet, cbnet -- emits a point estimate of ln MTT anyway, as
if the data fixed it, and the existing remedy (--detach-mtt) just severs its gradient. That throws
the parameter away rather than representing what is actually known about it.

This encoder splits its three outputs by how well the data constrain them:

  delay, ln CBV   identified   -> a point estimate, as before
  ln MTT          not          -> a *distribution*: the network emits a mean and a spread, and the
                                  decoder is handed a sample from it

That split is what the name refers to: the encoder is duo-headed by identifiability, not by task.

The consequence is the point. Reconstruction loss is averaged over the sampled MTT, so a pixel whose
curve genuinely pins MTT down is punished for a wide spread, while a pixel whose curve is consistent
with many MTTs pays nothing for admitting it. Nothing tells the network which pixels are which; the
Cramer-Rao geometry does, through the loss. Where the likelihood is flat, the spread opens for free.

Two further pieces follow from the same reasoning:

  * The spread is predicted from the cut length L. The bound is a function of how much washout was
    observed, so the head is given that explicitly rather than having to infer it.
  * The prior on the spread is one-sided. Collapsing to a point costs; being wide does not. This is
    the opposite of a VAE's KL term, which pulls spreads toward a fixed prior, and it is why this is
    not simply a VAE: there is no generative model and no sampling of the identified parameters.

The trunk is the U-Net of cnn_baseline, unchanged, so a comparison isolates the head.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class DuoNet(nn.Module):
    """U-Net trunk; the head emits a spread for the unidentified parameter alongside the means.

    Output layout matches the existing contract -- channels 0:4 means, 4:8 log-variances, 8: masks --
    so nothing downstream changes. The difference is that channel 4 (the log-variance of ln MTT) is
    now produced by its own head, conditioned on the cut, and is actually used.
    """

    def __init__(self, F, cin, cout, c=(48, 96, 192, 384), drop=0.2):
        super().__init__()
        self.enc = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2])])
        self.mid = block(c[2], c[3], drop)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                 nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([block(2 * c[2], c[2], drop), block(2 * c[1], c[1], drop),
                                  block(2 * c[0], c[0], drop)])
        self.head = nn.Conv2d(c[0], cout, 1)
        # the spread of ln MTT gets its own head, and sees the cut length the bound depends on
        self.spread = nn.Sequential(nn.Conv2d(c[0] + 1, c[0], 1), nn.SiLU(), nn.Conv2d(c[0], 1, 1))
        self.pool = nn.MaxPool2d(2)
        self.last_logs = None            # (B, 1, S, S), read by the loss after forward()

    def forward(self, x, cut_frac=None):
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
        if cut_frac is None:
            cut_frac = torch.ones(h.shape[0], 1, h.shape[2], h.shape[3], device=h.device, dtype=h.dtype)
        elif cut_frac.dim() == 1:
            cut_frac = cut_frac[:, None, None, None].expand(-1, 1, h.shape[2], h.shape[3])
        # a floor keeps the sampled MTT from ever collapsing to a hard point estimate
        self.last_logs = self.spread(torch.cat([h, cut_frac], 1)).clamp(-6.0, 1.5)
        return out


def net_duo(F, cin=None, cout=2):
    return DuoNet(F, cin or 4 * F, cout)


def sample_mtt(q, logs, training=True):
    """Replace the point ln MTT in q (P, 3) by a sample from N(mu, exp(logs)^2).

    q[:, 0] is ln MTT, q[:, 1] the delay, q[:, 2] ln CBV. Only the first is resampled: the other two
    are identified by the same curve and are left as point estimates. At eval time the mean is used,
    so the reported maps stay deterministic.
    """
    if not training:
        return q
    eps = torch.randn_like(q[:, 0])
    return torch.stack([q[:, 0] + eps * torch.exp(logs), q[:, 1], q[:, 2]], 1)


def spread_penalty(logs, target=0.0):
    """One-sided: collapsing the spread costs, widening it does not.

    A VAE's KL term would pull exp(logs) toward a fixed prior from both sides, which would fight the
    data wherever MTT *is* identified. Here only the collapse is penalised, so a pixel keeps a narrow
    spread exactly when its own curve earns one.
    """
    return torch.nn.functional.softplus(target - logs).mean()
