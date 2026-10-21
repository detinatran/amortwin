"""Generalisation of the probabilistic AmorTwin to other acquisition conditions (held-out ISLES'24 voxels).

python scripts/robustness.py <pairs.npz> <prob_weights_dir> <calib.json> <out.json> [--lengths full 30]

Every voxel is predicted by the model of its held-out fold (prob_amortwin.py weights), after one
perturbation of the inputs applied to the voxel curve, the patient's median tissue curve and the AIF:
  none                 the original inputs
  frames2s, frames3s   sampled every 2 or 3 s (the scan's own frames kept), linear interpolation back
                       to the 1 s grid of the network
  noise5, noise10, noise20   Gaussian noise of 5 / 10 / 20 % of the curve's peak added to the voxel curve
  late3s, late6s       the scan starts 3 or 6 s later (first frames dropped, shorter baseline)
  aif_early2, aif_late2   the AIF shifted by -2 / +2 s (a mistimed arterial input)
Targets stay the twin's parameters of the full scan. Per perturbation, length and target: Pearson r,
MAE, and the coverage of the 80 % interval with the conformal scale learned on the unperturbed data
(calib.json, not recalibrated).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from amortized_model import TMAX, AmorTwin, make_inputs

TARGETS = ("ln_rel_MTT", "delay_s", "ln_rel_CBV", "tmax_s")
PERT = ("none", "frames2s", "frames3s", "noise5", "noise10", "noise20", "late3s", "late6s", "aif_early2", "aif_late2")


def resample(x, step):
    """Keep every step-th frame and interpolate linearly back to 1 s (x: (N, T) tensor)."""
    T = x.shape[1]
    keep = torch.arange(0, T, step, device=x.device)
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    idx = torch.clamp(torch.searchsorted(keep.float(), t, right=True) - 1, 0, len(keep) - 2)
    t0, t1 = keep[idx].float(), keep[idx + 1].float()
    w = ((t - t0) / (t1 - t0)).clamp(0, 1)
    return x[:, keep[idx]] * (1 - w) + x[:, keep[idx + 1]] * w


def shift(x, s):
    """Shift curves by s frames (s > 0: later; zero-filled at the start, edge-held at the end)."""
    if s == 0:
        return x
    if s > 0:
        return torch.cat([torch.zeros_like(x[:, :s]), x[:, :-s]], 1)
    return torch.cat([x[:, -s:], x[:, -1:].expand(-1, -s)], 1)


def perturb(name, cur, ctx, aif, length, gen):
    if name.startswith("frames"):
        k = int(name[6])
        return resample(cur, k), resample(ctx, k), resample(aif, k), length
    if name.startswith("noise"):
        lvl = int(name[5:]) / 100.0
        peak = cur.amax(1, keepdim=True).clamp(min=1e-3)
        return cur + lvl * peak * torch.randn(cur.shape, generator=gen, device=cur.device), ctx, aif, length
    if name.startswith("late"):
        k = int(name[4])
        drop = lambda x: torch.cat([x[:, k:], torch.zeros_like(x[:, :k])], 1)  # noqa: E731
        return drop(cur), drop(ctx), drop(aif), torch.clamp(length - k, min=5)
    if name.startswith("aif_"):
        return cur, ctx, shift(aif, -2 if "early" in name else 2), length
    return cur, ctx, aif, length


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("calib", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--lengths", nargs="+", default=["full", "30"])
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.pairs)
    P = torch.tensor(z["patient_index"].astype(np.int64), device=dev)
    CUR = torch.tensor(z["curve"], dtype=torch.float16, device=dev)
    CTX = torch.tensor(z["ctx"], dtype=torch.float32, device=dev)
    AIF = torch.tensor(z["aif"], dtype=torch.float32, device=dev)
    LEN = torch.tensor(z["length"].astype(np.int64), device=dev)
    TIS = torch.tensor(z["tissue"].astype(np.int64), device=dev)
    Y = torch.tensor(z["target"], dtype=torch.float32, device=dev)
    FOLD = z["fold"].astype(np.int64)
    cal = json.loads(a.calib.read_text())
    K = len(TARGETS)
    models = {}
    for f in sorted(a.weights.glob("amortized_fold*.pt")):
        k = int(f.stem.replace("amortized_fold", ""))
        ck = torch.load(f, map_location=dev)
        m = AmorTwin(n_out=2 * K).to(dev)
        m.load_state_dict(ck["state"])
        models[k] = (m.eval(), ck["mu"].to(dev).float(), ck["sd"].to(dev).float())
    gen = torch.Generator(device=dev).manual_seed(0)
    out = {}
    for name in PERT:
        out[name] = {}
        for L in a.lengths:
            preds, sds, ys, covs = [], [], [], []
            for k, (m, mu, sd) in models.items():
                te = np.flatnonzero(FOLD == k)
                sc = torch.tensor([cal["scales"][L][t]["0.8"][str(k)] for t in TARGETS], device=dev)
                for s in range(0, len(te), 65536):
                    i = torch.as_tensor(te[s:s + 65536], device=dev)
                    cur, ctx, aif, ln = perturb(name, CUR[i].float(), CTX[P[i]], AIF[P[i]], LEN[i], gen)
                    if L != "full":
                        ln = torch.clamp(ln, max=int(L))
                    with torch.no_grad():
                        o = m(make_inputs(cur, ctx, aif, ln, TIS[i])).float()
                    mean = o[:, :K] * sd + mu
                    psd = torch.exp(0.5 * o[:, K:].clamp(-9.0, 6.0)) * sd
                    y = Y[i]
                    preds.append(mean.cpu()); ys.append(y.cpu())
                    covs.append(((y - mean).abs() <= 1.2816 * psd * sc).float().cpu())
            pr, yy, cv = torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(covs).numpy()
            out[name][L] = {t: {"r": float(np.corrcoef(pr[:, j], yy[:, j])[0, 1]), "mae": float(np.abs(pr[:, j] - yy[:, j]).mean()),
                                "cover80": float(cv[:, j].mean())} for j, t in enumerate(TARGETS)}
        print(name.ljust(11), "  ".join(f"{L}: " + " ".join(f"{t[:5]} r{out[name][L][t]['r']:.2f}/c{out[name][L][t]['cover80']:.2f}"
                                                         for t in TARGETS) for L in a.lengths), flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
