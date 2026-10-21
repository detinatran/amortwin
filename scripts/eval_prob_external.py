"""Probabilistic AmorTwin on an external cohort (no retraining): agreement with the cohort's own twin
fits, coverage with the ISLES'24 conformal scales, and after recalibration on a few patients of the
new cohort, for full and truncated scans.

python scripts/eval_prob_external.py <pairs.npz> <prob_weights_dir> <calib.json> <out.json>
       [--lengths full 25 30 35] [--m 10] [--repeats 200]

pairs.npz: export_pairs.py on the external cohort (targets = its twin fits on the full scans).
Prediction: the five ISLES'24 fold models as an ensemble, mean of the means and total variance
(mean of the variances + variance of the means). Coverage of the 80 % interval:
  raw          ensemble SD as is
  isles24      SD x the mean ISLES'24 conformal scale of the length and target (calib.json)
  recal_m      SD x a scale recalibrated on m patients of the cohort (the 80 % quantile of their
               z = |y - mean| / SD over 1.2816), tested on the other patients; mean over repeats
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from amortized_model import AmorTwin, make_inputs

TARGETS = ("ln_rel_MTT", "delay_s", "ln_rel_CBV", "tmax_s")
Z80 = 1.2816


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("calib", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--lengths", nargs="+", default=["full", "25", "30", "35"])
    p.add_argument("--m", type=int, default=10)
    p.add_argument("--repeats", type=int, default=200)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.pairs)
    P = torch.tensor(z["patient_index"].astype(np.int64), device=dev)
    CUR = torch.tensor(z["curve"], dtype=torch.float32, device=dev)
    CTX = torch.tensor(z["ctx"], dtype=torch.float32, device=dev)
    AIF = torch.tensor(z["aif"], dtype=torch.float32, device=dev)
    LEN = torch.tensor(z["length"].astype(np.int64), device=dev)
    TIS = torch.tensor(z["tissue"].astype(np.int64), device=dev)
    Y = z["target"].astype(np.float64)
    PID = z["pid"].astype(np.int64)
    cal = json.loads(a.calib.read_text())
    K = len(TARGETS)
    models = []
    for f in sorted(a.weights.glob("amortized_fold*.pt")):
        ck = torch.load(f, map_location=dev)
        m = AmorTwin(n_out=2 * K).to(dev)
        m.load_state_dict(ck["state"])
        models.append((m.eval(), ck["mu"].to(dev).float(), ck["sd"].to(dev).float()))
    rng = np.random.default_rng(0)
    pats = np.unique(PID)
    out = {"patients": int(len(pats)), "voxels": int(len(Y)), "lengths": {}}
    for L in a.lengths:
        means, vars_ = [], []
        for m, mu, sd in models:
            mm, vv = [], []
            for s in range(0, len(Y), 65536):
                i = torch.arange(s, min(s + 65536, len(Y)), device=dev)
                ln = LEN[i] if L == "full" else torch.clamp(LEN[i], max=int(L))
                with torch.no_grad():
                    o = m(make_inputs(CUR[i], CTX[P[i]], AIF[P[i]], ln, TIS[i])).float()
                mm.append((o[:, :K] * sd + mu).cpu().numpy())
                vv.append(((torch.exp(0.5 * o[:, K:].clamp(-9.0, 6.0)) * sd) ** 2).cpu().numpy())
            means.append(np.concatenate(mm))
            vars_.append(np.concatenate(vv))
        means, vars_ = np.stack(means), np.stack(vars_)
        mean = means.mean(0)
        sdv = np.sqrt(vars_.mean(0) + means.var(0))
        zz = np.abs(Y - mean) / sdv
        res = {}
        for j, t in enumerate(TARGETS):
            s24 = float(np.mean(list(cal["scales"][L][t]["0.8"].values())))
            covs = []
            for _ in range(a.repeats):
                c = rng.choice(pats, min(a.m, len(pats) - 1), replace=False)
                inc = np.isin(PID, c)
                s = float(np.quantile(zz[inc, j], 0.8)) / Z80
                covs.append(float(np.mean(zz[~inc, j] <= Z80 * s)))
            res[t] = {"r": float(np.corrcoef(mean[:, j], Y[:, j])[0, 1]), "mae": float(np.abs(mean[:, j] - Y[:, j]).mean()),
                      "cover80_raw": float(np.mean(zz[:, j] <= Z80)), "cover80_isles24": float(np.mean(zz[:, j] <= Z80 * s24)),
                      f"cover80_recal_m{a.m}": float(np.mean(covs)),
                      f"cover80_recal_m{a.m}_p10_p90": [float(np.percentile(covs, 10)), float(np.percentile(covs, 90))]}
        out["lengths"][L] = res
        print(L.rjust(4), " | ".join(f"{t[:7]} r {res[t]['r']:.2f} cov raw {res[t]['cover80_raw']:.2f} "
                                     f"i24 {res[t]['cover80_isles24']:.2f} recal {res[t][f'cover80_recal_m{a.m}']:.2f}" for t in TARGETS), flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
