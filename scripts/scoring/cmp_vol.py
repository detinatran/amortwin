"""Compare EPPINN and AmorTwin-S hypoperfused-volume error on the SAME ISLES'18 patients.

The two existing scorers disagree on the statistic -- agree18_json.py prints the MEAN absolute
error, score_eppinn_full.py the MEDIAN -- so quoting one beside the other would compare
different quantities. This prints both for both, on the intersection of the cases each method
actually produced, and reports n so a shrunken cohort cannot hide.

EPPINN gives no core (its maps carry no contralateral CBF reference), so only the
hypoperfused volume is comparable; core is deliberately not scored here.
"""
import os
# Paths are taken from AMORTWIN_ROOT (the directory holding the arm folders, cnn18/, cnn24/ and
# refs*.json), so these scorers run anywhere; the original runs used a machine-specific root.
ROOT = os.environ.get("AMORTWIN_ROOT", ".")
import json, sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk

CNN = Path(os.path.join(ROOT, "cnn18"))
EPP = Path("/tmp/eppinn_out_full")
REFS18 = os.path.join(ROOT, "cnn_scripts/refs18.json")

# --- EPPINN: Tmax > 6 s, voxel count * ml per voxel, exactly as score_eppinn_full.py ---
epp = {}
for d in sorted(EPP.glob("*/EPPINN/Tmax.nii.gz")):
    case = d.parent.parent.name
    npz = CNN / f"{case}.npz"
    if not npz.exists():
        continue
    tmax = sitk.GetArrayFromImage(sitk.ReadImage(str(d)))
    z = np.load(npz)
    valid = z["valid"].astype(bool)
    epp[case] = (float((tmax > 6)[valid].sum() * float(z["pixel_ml"])), float(z["hypo_ml"]))

# --- AmorTwin-S: probability-weighted volume at the 30 s cut ---
R = json.load(open(REFS18))
rh = {int(n): float(v) for n, v in R["hypo"].items()}
ats = {}
for arm in ("saeF_s0", "saeF_s1", "saeF_s2"):
    V = json.load(open(fos.path.join(ROOT, "{arm}/mvols18.json")))
    for name, d in V.items():
        pid = int(name)
        if pid in rh and "30" in d:
            ats.setdefault(pid, []).append(d["30"][1])

# EPPINN case names are training_N; map to the integer id the json uses
def pid_of(case):
    return int(case.split("_")[1])

common = sorted(p for c, p in ((c, pid_of(c)) for c in epp) if p in ats and p in rh)
print(f"n chung = {len(common)} bệnh nhân (EPPINN {len(epp)}, AT-S {len(ats)})\n")

def stats(errs):
    e = np.abs(np.array(errs))
    return e.mean(), np.median(e)

ep_err = [epp[f"training_{p}"][0] - rh[p] for p in common]
at_err = [np.mean(ats[p]) - rh[p] for p in common]

print(f"{'method':<14}{'mean |err|':>12}{'median |err|':>14}{'bias':>10}{'ca 0 ml':>10}")
m, md = stats(ep_err)
z = sum(1 for p in common if epp[f'training_{p}'][0] == 0)
print(f"{'EPPINN':<14}{m:>11.1f} {md:>13.1f} {np.mean(ep_err):>+9.1f}{z:>10}")
m, md = stats(at_err)
z = sum(1 for p in common if np.mean(ats[p]) == 0)
print(f"{'AmorTwin-S':<14}{m:>11.1f} {md:>13.1f} {np.mean(at_err):>+9.1f}{z:>10}")

# the five largest reference lesions -- where EPPINN was reported to fail
print("\n5 ca tổn thương lớn nhất (theo vendor):")
print(f"{'pid':>6}{'vendor':>9}{'EPPINN':>9}{'AT-S':>9}")
for p in sorted(common, key=lambda p: -rh[p])[:5]:
    print(f"{p:>6}{rh[p]:>9.1f}{epp[f'training_{p}'][0]:>9.1f}{np.mean(ats[p]):>9.1f}")
