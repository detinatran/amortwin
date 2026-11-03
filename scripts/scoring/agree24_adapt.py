"""Agreement for the test-time-adaptation arms, scored only on the half they did NOT adapt on.

fadapt_s{S}_h{H} adapts on half H of ISLES'24 (half24.json) and is therefore valid only on the
other half. Scoring all 94 patients, as agree18_json.py does, includes the adapted half and
inflates the arm -- exactly the mistake that would make this baseline look stronger than it is,
in a comparison whose point is that AmorTwin-S sees NO target data at all. Each seed's two halves
are complementary, so pooling h0's held-out half with h1's gives one full-cohort estimate per seed.

Reference refs24.json, volumes already probability-weighted.
"""
import os
# Paths are taken from AMORTWIN_ROOT (the directory holding the arm folders, cnn18/, cnn24/ and
# refs*.json), so these scorers run anywhere; the original runs used a machine-specific root.
ROOT = os.environ.get("AMORTWIN_ROOT", ".")
import sys, json, re
from pathlib import Path
import numpy as np

REFS18 = os.path.join(ROOT, "cnn_scripts/refs24.json")
HALF = os.path.join(ROOT, "cnn_scripts/half24.json")
LS = ("25", "30", "35", "full")


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


R = json.load(open(REFS18))
dec = {int(n): bool(v) for n, v in R["dec"].items()}
rc = {int(n): float(v) for n, v in R["core"].items()}
rh = {int(n): float(v) for n, v in R["hypo"].items()}
half = {int(k): int(v) for k, v in json.load(open(HALF)).items()}

print(f"{'arm':<18}{'n':>5}" + "".join(f"{L:>8}" for L in LS) + f"{'|core|30':>10}{'|hypo|30':>10}")

pooled = {}
for arg in sys.argv[1:]:
    p = Path(arg)
    m = re.search(r"_s(\d)_h(\d)$", p.name)
    if not m:
        print(f"{p.name}: not an adapt arm")
        continue
    seed, h = int(m.group(1)), int(m.group(2))
    V = {int(k): v for k, v in json.load(open(p / "mvols24.json")).items()}
    # held out = the half NOT adapted on
    pats = sorted(n for n in dec if n in V and half.get(n, -1) != h)
    row, e30 = [], (None, None)
    for L in LS:
        ok = 0; ec = []; eh = []
        for n in pats:
            c, hv = V[n][L]
            ok += defuse3(c, max(c, hv)) == dec[n]
            ec.append(abs(c - rc[n])); eh.append(abs(hv - rh[n]))
        row.append(100 * ok / len(pats))
        if L == "30":
            e30 = (float(np.mean(ec)), float(np.mean(eh)))
    print(f"{p.name:<18}{len(pats):>5}" + "".join(f"{v:>8.1f}" for v in row) +
          f"{e30[0]:>10.1f}{e30[1]:>10.1f}")
    pooled.setdefault(seed, []).extend(
        (n, V[n]) for n in pats)

print()
print("pooled per seed (h0's held-out half + h1's = the full cohort, each patient scored by the")
print("model that did not adapt on it):")
print(f"{'seed':<18}{'n':>5}" + "".join(f"{L:>8}" for L in LS) + f"{'|core|30':>10}{'|hypo|30':>10}")
agg = []
for seed in sorted(pooled):
    items = pooled[seed]
    row, e30 = [], (None, None)
    for L in LS:
        ok = 0; ec = []; eh = []
        for n, v in items:
            c, hv = v[L]
            ok += defuse3(c, max(c, hv)) == dec[n]
            ec.append(abs(c - rc[n])); eh.append(abs(hv - rh[n]))
        row.append(100 * ok / len(items))
        if L == "30":
            e30 = (float(np.mean(ec)), float(np.mean(eh)))
    print(f"{'seed '+str(seed):<18}{len(items):>5}" + "".join(f"{v:>8.1f}" for v in row) +
          f"{e30[0]:>10.1f}{e30[1]:>10.1f}")
    agg.append(row)
if len(agg) > 1:
    a = np.array(agg)
    print()
    print("mean +- SD over seeds: " + "  ".join(
        f"{L}={a[:,i].mean():.1f}+-{a[:,i].std(ddof=1):.1f}" for i, L in enumerate(LS)))
