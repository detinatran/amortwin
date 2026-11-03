"""Reverse-direction agreement on ISLES'24 from mvols24.json, over the whole cohort.

The reverse column of Table 2 is scored on all 145 ISLES'24 patients with a vendor DEFUSE 3
call -- not on held-out fold 3. agree_json.py restricts to fold 3 (n=30, always-YES 93.3 %),
which is the within-centre protocol and is NOT comparable to the reverse column: fold 3 is the
easiest fold, pooled always-YES over the full cohort is 91.3 %. Reference refs24.json.
"""
import os
# Paths are taken from AMORTWIN_ROOT (the directory holding the arm folders, cnn18/, cnn24/ and
# refs*.json), so these scorers run anywhere; the original runs used a machine-specific root.
ROOT = os.environ.get("AMORTWIN_ROOT", ".")
import sys, json
from pathlib import Path
import numpy as np

REFS = os.path.join(ROOT, "cnn_scripts/refs24.json")
LS = ("25", "30", "35", "full")


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


R = json.load(open(REFS))
dec = {int(n): bool(v) for n, v in R["dec"].items()}
rc = {int(n): float(v) for n, v in R["core"].items()}
rh = {int(n): float(v) for n, v in R["hypo"].items()}

first = True
rows = []
for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_dir():
        p = p / "mvols24.json"
    if not p.exists():
        print(f"{arg}: no mvols24.json"); continue
    V = {int(k): v for k, v in json.load(open(p)).items()}
    pats = sorted(n for n in dec if n in V)
    if not pats:
        print(f"{p.parent.name}: no overlap"); continue
    if first:
        y = sum(dec[n] for n in pats)
        print(f"ISLES'24: {len(pats)} patients, {y} YES / {len(pats)-y} NO, "
              f"always-YES = {100*y/len(pats):.1f} %")
        print()
        print(f"{'arm':<14}{'n':>5}" + "".join(f"{L:>8}" for L in LS) + f"{'|core|30':>10}{'|hypo|30':>10}")
        first = False
    row, e30 = [], (None, None)
    for L in LS:
        ok = 0; ec = []; eh = []
        for n in pats:
            c, h = V[n][L]
            ok += defuse3(c, max(c, h)) == dec[n]
            ec.append(abs(c - rc[n])); eh.append(abs(h - rh[n]))
        row.append(100 * ok / len(pats))
        if L == "30":
            e30 = (float(np.mean(ec)), float(np.mean(eh)))
    print(f"{p.parent.name:<14}{len(pats):>5}" + "".join(f"{v:>8.1f}" for v in row) +
          f"{e30[0]:>10.1f}{e30[1]:>10.1f}")
    rows.append(row)
if len(rows) > 1:
    a = np.array(rows)
    print()
    print("mean +- SD: " + "  ".join(
        f"{L}={a[:,i].mean():.1f}+-{a[:,i].std(ddof=1):.1f}" for i, L in enumerate(LS)))
