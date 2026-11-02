"""FloorNet: stop the volume accumulating a probability floor, by making the encoder commit.

WHY THIS, AFTER THREE OF MY OWN DESIGNS FAILED

Nine arms, grouped by where they intervene, against the U-Net's |core| of 8.6 ml at 30 s:

    intervention       n   mean |core|   mean agreement   arms
    nothing            1       8.6 ml        93.3 %       U-Net
    encoder (input)    3       7.4           92.8         TSNet, CBNet, DuoNet-2
    encoder (head)     2       7.5           93.3         DuoNet, DuoNet-3
    readout + loss     2      11.6           88.3         CalNet, GapNet
    decision loss      1      19.8           73.3         MarginNet

Every arm that touched the readout or the decision made the volume **worse**; every arm that
touched only the encoder kept it at or below the U-Net. No exceptions in nine arms. My three
proposals -- MarginNet, GapNet and the gap bias knob I discarded at the simulation stage -- were
all readout interventions, and all three failed. So this one changes the encoder only, and the
readout stays exactly `V = sum_v p(v) w(v)`, unthresholded, as the paper requires.

WHAT THE ERROR ACTUALLY IS

Decomposing the predicted core volume by probability band (U-Net s2, 30 s, mean per patient):

    p band        ml/patient   % of volume   voxels/patient
    [0.00,0.02)       3.02        16.5 %        63 328
    [0.02,0.05)       1.82        10.0 %         4 187
    [0.05,0.10)       1.61         8.8 %         1 591
    [0.10,0.30)       3.40        18.6 %         1 367
    [0.30,0.50)       2.47        13.5 %           435
    [0.50,1.00]       5.96        32.6 %           503
    total            18.28                     (reference 16.07)

**63 328 voxels the network has all but rejected (p < 0.02) contribute 3.02 ml, and the whole
p < 0.10 region contributes 35 % of the volume.** The bias is +2.21 ml, and zeroing just the
p < 0.02 band would leave -0.81 ml -- i.e. that floor is most of the error. This is not a decision
error and not a boundary error: checked directly, false mass more than 5 voxels from the reference
at p > 0.5 is only 0.67 ml per patient, and **100 % of it lies in the lesion's own hemisphere**, so
the network is not confusing sides. The error is a thin, diffuse probability spread over the whole
brain that the weighted sum integrates into a systematic offset.

It also explains the size dependence that made relative error ~1.0 on small lesions and ~0.35 on
large ones (24 of the 30 patients have a reference core under 20 ml): a fixed ~3 ml floor is a
third of an 8 ml lesion and a fifteenth of a 47 ml one.

WHY A THRESHOLD IS NOT THE ANSWER

Zeroing p < 0.02 at readout would fix the core, but thresholding the volume is exactly what this
paper rules out (it was one of four mechanisms measured to lose 30-62 points), it is not
differentiable, and the same floor on the hypo channel would cut far more true volume. The floor
has to stop being produced, not be removed afterwards.

THE MECHANISM

A confidence penalty on the near-zero region, applied during training to the mask logits:

    for voxels with p < tau, add  w * p

A plain L1 on p over the whole brain would also shrink the true lesion. Restricting it to the band
the network has already rejected makes it a *commitment* pressure: a voxel the encoder is unsure
about (p = 0.3) is untouched, a voxel it has effectively rejected (p = 0.01) is pushed to 0 instead
of being left to contribute a hundredth of a voxel each, 63 000 times over. The gradient of `w * p`
is constant in the band, so it does not fight the cross-entropy where the cross-entropy is strong.

Kept from what was measured to work: the unchanged `cnn_baseline` trunk, and CalNet's per-channel
learned temperature -- the single mechanism in the table measured to fix a volume (112.9 against
the vendor's 112.4 ml), kept even though CalNet's agreement was poor, because its *volume* claim
held and its failure was traced to the decisiveness penalty, which is not used here.

Deliberately excluded, each for a measured reason:
  * any loss on the decision or on a volume (MarginNet |core| 19.8, GapNet 12.6, CalNet 10.7 --
    all worse than the U-Net's 8.6);
  * CalNet's decisiveness penalty -- drove the core to 0 ml on p70 and p144;
  * a temporal front end (TSNet, DuoNet-2) -- 14-65x the cost, all gaps inside seed noise;
  * a learned spread on ln MTT (DuoNet, DuoNet-3) -- the trained probe showed it flat in L.

HONEST LIMIT, stated before the run

The claim is about volume, not agreement: fold 3 has 28 YES of 30, so always-YES scores 93.3 %,
the best arm leaves one patient of headroom, and two seeds need a 1.0-patient gap to clear the
measured seed noise. What is testable on 30 patients is whether the low-probability mass and the
volume bias fall -- both measured over ~63 000 voxels per patient, not over 2 decisions. If the
mass falls and the bias does not, the floor was not the cause and this design is wrong too.
"""
import torch
import torch.nn as nn


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
    if drop:
        layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class FloorNet(nn.Module):
    """The cnn_baseline trunk and CalNet's temperature. The mechanism is in the loss, not here.

    The architecture is deliberately the U-Net plus two scalars: the measurement says the volume
    error is a property of how the probabilities are distributed, not of what the encoder can see,
    so adding capacity would confound the test.
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
        self.log_temp = nn.Parameter(torch.zeros(2))   # calnet's, the one measured to fix a volume

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
            t = torch.exp(self.log_temp).clamp(0.2, 5.0)[None, :, None, None]
            out = torch.cat([out[:, :8], out[:, 8:10] / t], 1)
        return out


def net_floor(F, cin=None, cout=2):
    return FloorNet(F, cin or 4 * F, cout)


def floor_penalty(logits, valid, tau=0.05):
    """Mean probability of the voxels the network has already rejected (p < tau), inside the brain.

    63 328 voxels per patient sit below p = 0.02 and contribute 3.02 of 18.28 ml, so the weighted
    sum reads a floor that no single voxel is responsible for. Penalising `p` only where `p < tau`
    pushes that floor to zero while leaving every voxel the encoder is genuinely unsure about
    (p = 0.3, which carries 18.6 % of the volume and is where the lesion boundary lives) untouched.

    The mask is detached: the band membership is a selection, not a quantity to differentiate
    through, so the gradient is a constant pressure inside the band and exactly zero outside it.
    """
    p = torch.sigmoid(logits)
    v = valid.float()
    sel = ((p < tau).float() * v).detach()
    return (p * sel).sum() / sel.sum().clamp_min(1.0)
