"""Reverse-direction agreement on ISLES'24 from thr24.json (the mvols_thr.py layout).

cei_tabbase.sh runs mvols_thr.py, which writes thr24.json rather than the mvols24.json that
agree24_json.py expects, so the trunk arms looked unscored. The probability-weighted volumes
the paper uses are the "exp" pair [core, hypo] -- the same V = sum_v p(v) w(v) as everywhere
else; the "thr" entries are thresholded readouts and are NOT what Table 2 reports.

Reference refs24.json, all 145 patients, as agree24_json.py uses.
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
for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_dir():
        p = p / "thr24.json"
    if not p.exists():
        print(f"{arg}: no thr24.json")
        continue
    V = json.load(open(p))
    if first:
        common = [int(n) for n in V if int(n) in dec]
        yes = sum(dec[n] for n in common)
        print(f"ISLES'24: {len(common)} patients, {yes} YES / {len(common)-yes} NO, "
              f"always-YES = {100*yes/len(common):.1f} %")
        print(f"\n{'arm':<18}{'n':>5}{'25':>8}{'30':>8}{'35':>8}{'full':>8}"
              f"{'|core|30':>10}{'|hypo|30':>10}")
        first = False
    row, ce, he, n = [], [], [], 0
    for L in LS:
        ok = tot = 0
        for name, d in V.items():
            pid = int(name)
            if pid not in dec or L not in d:
                continue
            core, hypo = d[L]["exp"]
            union = max(core, hypo)
            ok += (defuse3(core, union) == dec[pid]); tot += 1
            if L == "30":
                ce.append(abs(core - rc[pid])); he.append(abs(hypo - rh[pid]))
        row.append(100 * ok / tot if tot else float("nan"))
        if L == "25":
            n = tot
    print(f"{p.parent.name:<18}{n:>5}" + "".join(f"{v:>8.1f}" for v in row) +
          f"{np.median(ce):>10.1f}{np.median(he):>10.1f}")
