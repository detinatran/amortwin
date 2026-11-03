"""Agreement on ISLES'24 fold 0 -- the held-out fold of the ported 4D mJ-Net.

The study's other within-centre scorer uses fold 3; mJNet-4D was trained on fold 0, so it must
be scored on fold 0 or the numbers are not comparable to its own training split. Reference is
refs24.json, volumes are probability-weighted, as everywhere.
"""
import os
# Paths are taken from AMORTWIN_ROOT (the directory holding the arm folders, cnn18/, cnn24/ and
# refs*.json), so these scorers run anywhere; the original runs used a machine-specific root.
ROOT = os.environ.get("AMORTWIN_ROOT", ".")
import sys, json
from pathlib import Path
import numpy as np

D = Path(ROOT)
REFS = D / "cnn_scripts/refs24.json"
FOLDS = D / "cnn_scripts/learned_folds24.json"
LS = ("25", "30", "35", "full")
FOLD = 0


def defuse3(core, union):
    return bool(core < 70 and (union / core >= 1.8 if core > 0 else True) and union - core >= 15)


R = json.load(open(REFS))
dec = {int(n): bool(v) for n, v in R["dec"].items()}
rc = {int(n): float(v) for n, v in R["core"].items()}
rh = {int(n): float(v) for n, v in R["hypo"].items()}
folds = {int(k): v for k, v in json.loads(FOLDS.read_text()).items()}
test = sorted(n for n in dec if folds.get(n, -1) == FOLD)
y = sum(dec[n] for n in test)
print(f"ISLES'24 fold {FOLD}: {len(test)} bệnh nhân, {y} YES / {len(test)-y} NO, "
      f"always-YES = {100*y/len(test):.1f} %")
print()
print(f"{'arm':<14}{'n':>4}" + "".join(f"{L:>8}" for L in LS) + f"{'|core|30':>10}{'|hypo|30':>10}")

for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_dir():
        p = p / "mvols24.json"
    V = {int(k): v for k, v in json.load(open(p)).items()}
    pats = [n for n in test if n in V]
    if not pats:
        print(f"{p.parent.name}: không khớp bệnh nhân nào"); continue
    row, e30 = [], (0.0, 0.0)
    for L in LS:
        ok = 0; ec = []; eh = []
        for n in pats:
            c, h = V[n][L]
            ok += defuse3(c, max(c, h)) == dec[n]
            ec.append(abs(c - rc[n])); eh.append(abs(h - rh[n]))
        row.append(100 * ok / len(pats))
        if L == "30":
            e30 = (float(np.mean(ec)), float(np.mean(eh)))
    print(f"{p.parent.name:<14}{len(pats):>4}" + "".join(f"{v:>8.1f}" for v in row) +
          f"{e30[0]:>10.1f}{e30[1]:>10.1f}")
