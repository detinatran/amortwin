"""Paired bootstrap CI for symbolic vs learned decoder, the comparison the review asks for.

The paper already reports bootstrap CIs of each arm against the CONTROL; what it does not give
is a direct interval between the two decoders, which is the comparison its claim rests on.

Resampling is over PATIENTS, as everywhere else in the paper: both arms score the same cohort,
so the pairing is per patient. Each arm's agreement at a patient is averaged over its seeds
first (LD has five, AT-S three here), so the interval reflects patient variation, not seed
variation -- the seed spread is reported separately as the subscripts in Table 2.
"""
import os
# Paths are taken from AMORTWIN_ROOT (the directory holding the arm folders, cnn18/, cnn24/ and
# refs*.json), so these scorers run anywhere; the original runs used a machine-specific root.
ROOT = os.environ.get("AMORTWIN_ROOT", ".")
import json, sys
from pathlib import Path
import numpy as np

D = Path(ROOT)
REFS18 = D / "cnn_scripts/refs24.json"
LS = ("25", "30", "35")
RNG = np.random.default_rng(0)
B = 10000


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


R = json.load(open(REFS18))
dec = {int(n): bool(v) for n, v in R["dec"].items()}


def hits(dirs, L):
    """Per-patient mean correctness over an arm's seeds."""
    per = {}
    for d in dirs:
        V = {int(k): v for k, v in json.load(open(D / d / "mvols24.json")).items()}
        for n in dec:
            if n in V:
                c, h = V[n][L]
                per.setdefault(n, []).append(defuse3(c, max(c, h)) == dec[n])
    return {n: float(np.mean(v)) for n, v in per.items()}


LD = [f"rev_ld_s{s}" for s in range(3)]
# Table 2 reports ten seeds for AT-S in reverse (see its caption), so the interval uses all ten.
ATS = [f"rsaeF_s{s}" for s in range(10)]

print(f"{'cut':>5}  {'AT-S':>6}  {'LD':>6}  {'diff':>6}   95 % CI of the paired difference")
for L in LS:
    a, b = hits(ATS, L), hits(LD, L)
    pats = sorted(set(a) & set(b))
    da = np.array([a[n] for n in pats])
    db = np.array([b[n] for n in pats])
    d = da - db
    idx = RNG.integers(0, len(pats), size=(B, len(pats)))
    boot = 100 * d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{L:>5}  {100*da.mean():6.1f}  {100*db.mean():6.1f}  {100*d.mean():+6.1f}   "
          f"[{lo:+.1f}, {hi:+.1f}]{'  excludes zero' if lo > 0 or hi < 0 else ''}   n={len(pats)}")
