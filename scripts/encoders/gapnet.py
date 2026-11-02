"""GapNet: predict the gap that DEFUSE 3 reads, and how uncertain that gap is.

Every choice below is forced by a measurement, and two of them come from a design I had to throw
away. The record of that matters, so it is kept here.

WHAT THE MEASUREMENTS SAY (held-out fold 3, pooled over ten trained arms, 10 x 30 decisions)

1.  **79 % of all wrong decisions flip the mismatch clause** `max(core,hypo) - core >= 15 ml`;
    21 % flip `core < 70`; **0 % flip the ratio clause**. The decisive quantity is the gap
    `hypo - core`. The ratio needs no attention at all.

2.  **The error lives in hypo, not core**: signed bias +0.40 ml on core against **+12.18 ml on
    hypo**; mean |error| 8.3 vs 31.8 ml. Capacity spent on core is spent where nothing is to win.

3.  **The two heads' errors are correlated, r = +0.613**, so the gap's error (28.4 ml) is smaller
    than if they were independent (32.9 ml) -- the biases partly cancel in the subtraction. A
    design that calibrates each head separately destroys that cancellation.

4.  **64 % of the hypo volume comes from undecided voxels** (p in 0.05..0.8 gives ~72 of ~136 ml
    per patient) while the whole error is +29 ml. The readout is dominated by voxels the network
    is unsure of.

5.  **CalNet already fixed the volume** (112.9 ml against the vendor's 112.4) with a learned
    temperature **and still scored the lowest agreement of any arm, 90.0 %**. Calibrating the
    volume is not the same as getting the decision right.

WHAT I TRIED FIRST, AND WHY IT IS NOT HERE

The obvious reading of (1)-(3) is: calibrate the gap with one signed scalar and a relative loss on
it. I implemented that and simulated it with the measured error structure. It made the decision
**worse**: clause flips went 10.6 % -> 18.4 %, because a relative loss is dominated by the large
gaps (the reference gaps run 4..289 ml) and dragged the shift to -20 ml, wrecking the only region
that matters, near 15 ml. Sweeping the shift by hand then showed the best constant was **+15 ml**
(flips 11.4 % -> 7.8 %) -- i.e. inflate every gap until almost everyone clears the clause. That is
not a correction, it is re-learning the YES prior of a cohort that is 28/30 YES. Only 10.6 % of
patients sit within 10-20 ml of the clause, so any global shift trades away correct
far-from-boundary calls to buy nothing. **A bias knob on the gap is the wrong mechanism**, and no
amount of tuning rescues it.

WHAT THE EVIDENCE SUPPORTS INSTEAD

Flips do not concentrate where the gap is large; they concentrate where the gap is uncertain
*relative to its distance from the clause*. With the measured heteroscedastic error structure
(gap sd ~ 8 + 0.25 x gap):

    margin = |predicted gap - 15| / sd(gap)      flip rate
            0.00 - 0.25                            53.0 %
            0.25 - 0.50                            31.2 %
            0.50 - 1.00                            22.6 %
            1.00 - 2.00                             8.7 %
            > 2.00                                  0.8 %

A 66x spread, and the margin is computable at inference with no reference. So GapNet's job is not
to shift the gap but to **say how certain the gap is**: it predicts the gap's standard deviation
per slice, trained by Gaussian likelihood on the gap. That turns the readout into a decision with
a margin, which is what the clause-based rule needs and what no arm in the table produces.

Abstaining on the smallest-margin cases converts that into accuracy where it is claimed:
abstain 10 % -> 96.0 %, 20 % -> 97.5 %, 30 % -> 98.7 % (from 92.9 % at full coverage). Reporting
accuracy at a stated coverage is an honest claim on 30 patients; beating 93.3 % by one patient is
not.

COMPOSITION, and what is deliberately excluded

Kept: the unchanged `cnn_baseline` U-Net trunk; CalNet's per-channel learned temperature, which is
the one mechanism measured to fix the volume (112.9 vs 112.4 ml).

Added: a small head that predicts log sd of the gap per slice, and a Gaussian NLL on the gap. Two
scalars plus one 1x1 conv.

Excluded, each for a measured reason:
  * a signed bias knob on the gap -- the simulation above: 10.6 % -> 18.4 % flips;
  * CalNet's decisiveness penalty -- drove the core to 0 ml on p70 and p144, making the ratio
    meaningless (80.5) and flipping the decision;
  * a temporal front end (TSNet, DuoNet-2) -- 14-65x the cost, all gaps inside seed noise;
  * a learned spread on ln MTT (DuoNet, DuoNet-3) -- the trained probe showed it flat in L
    (-2.379/-2.436/-2.437/-2.447), so the mechanism never engaged;
  * a smooth surrogate of the whole rule (MarginNet) -- 73-80 % agreement, |core| 19.8 vs 8.3 ml:
    training on the decision wrecks the volumes.

HONEST LIMIT, stated before any result

Fold 3 has 30 patients, 28 YES, so always-YES scores 93.3 %; the best arm leaves one patient of
headroom at 30 s, and two seeds per arm need a 1.0-patient gap to clear the measured seed noise.
GapNet cannot be shown to beat the table on this fold, and the abstention numbers above are from
simulation with the measured error structure, not from ISLES. What is testable on 30 patients is
whether the predicted sd ranks the errors -- a correlation, not a 2-patient difference.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class GapNet(nn.Module):
    """The cnn_baseline trunk, CalNet's temperature, and a head for the gap's uncertainty.

    The temperature is per channel because core and hypo are not equally miscalibrated (+0.40
    against +12.18 ml). The uncertainty head is pooled to one number per slice because the gap is
    a per-slice quantity: it is a sum over voxels, and what the clause reads is that sum.
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
        self.pool = nn.MaxPool2d(2)
        self.cout = cout
        self.log_temp = nn.Parameter(torch.zeros(2))     # calnet's, the one measured to work
        # log sd of the gap, per slice: a 1x1 conv read out by global pooling
        self.gap_sd = nn.Conv2d(c[0], 1, 1)
        self.log_sd_bias = nn.Parameter(torch.tensor(2.5))   # ~12 ml, the measured scale
        self.last_log_sd = None

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
        # one log sd per slice; the gap is a sum over voxels, so the uncertainty is per slice too
        self.last_log_sd = (self.gap_sd(h).mean(dim=(1, 2, 3)) + self.log_sd_bias).clamp(-2.0, 7.0)
        if self.cout >= 10:
            t = torch.exp(self.log_temp).clamp(0.2, 5.0)[None, :, None, None]
            out = torch.cat([out[:, :8], out[:, 8:10] / t], 1)
        return out


def net_gap(F, cin=None, cout=2):
    return GapNet(F, cin or 4 * F, cout)


def gap_nll(logits, target, valid, log_sd, pixel_ml=1.0):
    """Gaussian NLL of the reference gap under the predicted gap and its predicted sd.

    This is a likelihood, not a bias correction: it does not push the gap anywhere, it asks the
    network to state how far off the gap might be. A simulated bias knob on the same quantity made
    the decision worse (flips 10.6 % -> 18.4 %), whereas the predicted sd separates flip rates by
    66x (53 % at margin < 0.25 against 0.8 % above 2). The term is scale-free in sd, so it cannot
    be minimised by inflating the uncertainty: the log_sd penalty balances the residual.
    """
    p = torch.sigmoid(logits)
    v = valid.float()
    pc = (p[:, 0:1] * v).flatten(1).sum(-1) * pixel_ml
    ph = (p[:, 1:2] * v).flatten(1).sum(-1) * pixel_ml
    tc = (target[:, 0:1] * v).flatten(1).sum(-1) * pixel_ml
    th = (target[:, 1:2] * v).flatten(1).sum(-1) * pixel_ml
    pg = torch.clamp(ph - pc, min=0.0)
    tg = torch.clamp(th - tc, min=0.0)
    sd = torch.exp(log_sd).clamp(1e-2, 1e3)
    return (0.5 * ((pg - tg) / sd) ** 2 + log_sd).mean()


def decision_margin(core_ml, hypo_ml, sd_ml, thr=15.0):
    """|gap - 15| / sd: the abstention signal, computable without any reference."""
    gap = torch.clamp(hypo_ml - core_ml, min=0.0)
    return (gap - thr).abs() / torch.as_tensor(sd_ml).clamp(min=1e-2)
