"""Agreement on ISLES'18 from mvols18.json, for arms whose pred18/ the runner deleted.

Both learned-decoder runners gather the per-patient volumes into mvols18.json and then
`rm -rf pred18`, so neither agree18.py (wants pred18/) nor agree_cei.py (wants pred/) can
read these arms. Scoring the json against refs24.json, as agree_json.py does, is worse than
useless here: this is the forward direction, the keys are ISLES'18 ids 1-94, and the '24 ids
collide with 15 of them, so it silently scored 15 wrong patients and reported 20 % agreement
with a 127 ml hypo error. Reference is refs18.json and all 94 patients, as agree18.py uses.

Volumes in the json are already probability-weighted sums, never thresholded.
"""
import os
# Paths are taken from AMORTWIN_ROOT (the directory holding the arm folders, cnn18/, cnn24/ and
# refs*.json), so these scorers run anywhere; the original runs used a machine-specific root.
ROOT = os.environ.get("AMORTWIN_ROOT", ".")
import sys, json
from pathlib import Path
import numpy as np

REFS18 = os.path.join(ROOT, "cnn_scripts/refs18.json")
LS = ("25", "30", "35", "full")


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


R = json.load(open(REFS18))
dec = {int(n): bool(v) for n, v in R["dec"].items()}
rc = {int(n): float(v) for n, v in R["core"].items()}
rh = {int(n): float(v) for n, v in R["hypo"].items()}

first = True
for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_dir():
        p = p / "mvols18.json"
    if not p.exists():
        print(f"{arg}: no mvols18.json")
        continue
    V = {int(k): v for k, v in json.load(open(p)).items()}
    pats = sorted(n for n in dec if n in V)
    if not pats:
        print(f"{p.parent.name}: no overlap with refs18")
        continue
    if first:
        y = sum(dec[n] for n in pats)
        print(f"ISLES'18: {len(pats)} patients, {y} YES / {len(pats)-y} NO, "
              f"always-YES = {100*y/len(pats):.1f} %  (always-NO = {100-100*y/len(pats):.1f} %)")
        print()
        print(f"{'arm':<16}" + "".join(f"{L:>8}" for L in LS) + f"{'|core|30':>10}{'|hypo|30':>10}")
        first = False
    row = []
    e30 = (None, None)
    for L in LS:
        ok = 0
        ec, eh = [], []
        for n in pats:
            c, h = V[n][L]
            # the json stores [core, hypo]; DEFUSE-3 takes the union as max(core, hypo)
            ok += defuse3(c, max(c, h)) == dec[n]
            ec.append(abs(c - rc[n]))
            eh.append(abs(h - rh[n]))
        row.append(100 * ok / len(pats))
        if L == "30":
            e30 = (float(np.mean(ec)), float(np.mean(eh)))
    print(f"{p.parent.name:<16}" + "".join(f"{v:>8.1f}" for v in row) +
          f"{e30[0]:>10.1f}{e30[1]:>10.1f}")
