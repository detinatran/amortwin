"""Physics-consistency test-time adaptation of AmorTwin-2D: one patient at a time, no data of the target centre.

python scripts/at2d_tta.py <cnn_dir> <x_dir> <model_dir> <out.json> --loss physics|entropy|none [--steps 10] [--lr 1e-3]

For every patient and every cut length L (25, 30, 35 s, full) the GroupNorm affine parameters of the source model
(at2d_fold0.pt of model_dir, the model trained on all patients of the source centre) are adapted from scratch
(episodic: reset for every patient and length), with the dropout off, by Adam on one loss computed on the cut scan only:
  physics: the observed part of every tissue curve (frames < L) is rebuilt from the predicted twin quantities with
           the indicator-dilution model of the twin with its dispersion at the lower bound (Tmax = delay + 0.05 s ln(..)):
               c(t) = A exp(l_cbv - l_mtt) sum_{s <= t} a(s - delay) exp(-(t - s) / (M0 exp(l_mtt)))
           where a is the cut arterial input, l_mtt / l_cbv the predicted log MTT / CBV relative to the patient's median
           and A, M0 two per-patient scalars (amplitude, median MTT), first fitted with the network frozen (grid over M0,
           closed-form A) and then adapted with it. The curves are smoothed in time as for the twin (Gaussian, 1.8
           frames, on the cut part only). The loss is the twin's floored nMSE: per pixel the squared error over
           the observed frames divided by max(curve energy, 5 % of the mean energy), averaged over valid pixels; frames
           whose delayed arterial sample lies at or beyond the cut are left out.
  entropy: TENT, the binary entropy of the two vendor-mask outputs averaged over valid pixels (needs no physics output).
  none:    no adaptation (checks the pipeline against the stored predictions).
Output: {patient: {L: [core ml, hypo ml]}} from the mask outputs, as mvols_dir.py, plus the physics loss before/after.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amortwin2d import inputs, load  # noqa: E402
from cnn_baseline import LENGTHS, net  # noqa: E402

M0_GRID = np.arange(2.0, 16.01, 1.0)
SMOOTH = 1.8                     # frames, the twin's smooth_sigma_frames


def smooth_cut(ctc, L):
    """The curves (P, F) cut at L and smoothed in time like the twin's (frames < L only, no look-ahead past the cut)."""
    import torch
    c = ctc[:, :L].cpu().numpy()
    return torch.as_tensor(gaussian_filter1d(c, SMOOTH, axis=1, mode="nearest"), device=ctc.device)


def shifted_aif(aif, delay, L):
    """a(t - delay) for t < L by linear interpolation of the cut arterial input aif (F,); also the frames whose
    sample lies inside [0, L - 1]. delay (P,) -> (P, L), (P, L) bool."""
    import torch
    t = torch.arange(L, dtype=aif.dtype, device=aif.device)[None, :] - torch.nan_to_num(delay)[:, None]
    ok = t <= L - 1
    tc = t.clamp(0, L - 1)
    i0 = tc.floor().long().clamp(max=L - 2) if L > 1 else tc.floor().long()
    f = tc - i0
    a = aif[:L]
    v = a[i0] * (1 - f) + a[(i0 + 1).clamp(max=L - 1)] * f
    return torch.where(t < 0, torch.zeros_like(v), v), ok


def rebuild(aif, q, L, m0):
    """Unit-amplitude model curves (P, L) of the pixels with quantities q (P, 3) = (l_mtt, delay, l_cbv)."""
    import torch
    a, ok = shifted_aif(aif, q[:, 1], L)
    decay = torch.exp(-1.0 / (m0 * torch.exp(q[:, 0])))
    y, out = torch.zeros_like(q[:, 0]), []
    for k in range(L):
        y = y * decay + a[:, k]
        out.append(y)
    return torch.exp(q[:, 2] - q[:, 0])[:, None] * torch.stack(out, 1), ok


def phys_loss(model_out, nz, c, pix, aif, L, A, m0):
    """Floored nMSE of the observed tissue curves c (P, L) of the valid pixels pix."""
    import torch
    L = c.shape[1]                 # the stored curves stop at F frames even when the scan is longer
    mu, sd = nz
    q =(model_out[:, :3] * sd[:3, None, None] + mu[:3, None, None]).permute(0, 2, 3, 1)[pix]
    y, ok = rebuild(aif, q, L, m0)
    e = (c ** 2).sum(1)
    den = torch.clamp(e, min=0.05 * float(e.mean()))
    return ((ok * (A * y - c) ** 2).sum(1) / den).mean(), y, ok


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("cnn")
    p.add_argument("x")
    p.add_argument("model", type=Path)
    p.add_argument("out")
    p.add_argument("--loss", choices=("physics", "entropy", "none"), required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patients", help="comma-separated subset (smoke test)")
    a = p.parse_args()
    dev = "cuda"
    data = load(a.cnn, a.x)
    if a.patients:
        data = {n: data[n] for n in map(int, a.patients.split(","))}
    F = next(iter(data.values()))["ctc"].shape[1]
    model = net(F, cin=4 * F, cout=10).to(dev).eval()
    state = torch.load(a.model / "at2d_fold0.pt", map_location=dev)
    nzf = np.load(a.model / "at2d_fold0_norm.npz")
    nz = tuple(torch.as_tensor(nzf[k], dtype=torch.float32, device=dev) for k in ("mu", "sd"))
    gn = [m for m in model.modules() if isinstance(m, torch.nn.GroupNorm)]
    res, t0 = {}, time.time()
    for n, d in data.items():
        v = torch.as_tensor(d["valid"].astype(np.float32), device=dev)
        pix = v > 0.5
        ctc = torch.as_tensor(d["ctc"].astype(np.float32), device=dev).permute(0, 2, 3, 1)[pix]   # (P, F)
        aif = torch.as_tensor(d["aif"], dtype=torch.float32, device=dev)
        rec = {}
        for Ls in LENGTHS:
            L = d["frames"] if Ls == "full" else int(Ls)
            model.load_state_dict(state)
            X = torch.as_tensor(inputs(d, L)).to(dev)
            c, aL = smooth_cut(ctc, L), aif.clone()
            aL[L:] = 0
            with torch.no_grad():          # per-patient scalars with the network frozen: grid over M0, closed-form A
                o = model(X)
                best = None
                for m0 in M0_GRID:
                    _, y, ok = phys_loss(o, nz, c, pix, aL, L, 1.0, float(m0))
                    A = float((ok * y * c).sum() / (ok * y * y).sum().clamp_min(1e-12))
                    lo = float(phys_loss(o, nz, c, pix, aL, L, A, float(m0))[0])
                    if best is None or lo < best[0]:
                        best = (lo, A, float(m0))
            l0, A0, M0 = best
            l1 = l0
            if a.loss != "none":
                logA = torch.tensor(float(np.log(max(A0, 1e-12))), dtype=torch.float32, device=dev, requires_grad=True)
                logM = torch.tensor(float(np.log(M0)), dtype=torch.float32, device=dev, requires_grad=True)
                params = [q for m in gn for q in (m.weight, m.bias)]
                for q in model.parameters():
                    q.requires_grad_(False)
                for q in params:
                    q.requires_grad_(True)
                opt = torch.optim.Adam(params + ([logA, logM] if a.loss == "physics" else []), lr=a.lr)
                for _ in range(a.steps):
                    o = model(X)
                    if a.loss == "physics":
                        loss = phys_loss(o, nz, c, pix, aL, L, torch.exp(logA), torch.exp(logM))[0]
                    else:
                        pr = torch.sigmoid(o[:, 8:10])
                        ent = -(pr * torch.log(pr + 1e-6) + (1 - pr) * torch.log(1 - pr + 1e-6))
                        loss = (ent * v[:, None]).sum() / v.sum().clamp_min(1) / 2
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    o = model(X)
                    l1 = float(phys_loss(o, nz, c, pix, aL, L, float(torch.exp(logA)), float(torch.exp(logM)))[0]) \
                        if a.loss == "physics" else l0
            pm = torch.sigmoid(o[:, 8:10]).detach()
            vol = [float((pm[:, j] * v).sum()) * d["pixel_ml"] for j in (0, 1)]
            rec[Ls] = vol
            rec[f"phys_{Ls}"] = [l0, l1, M0]
        res[n] = rec
        print(n, {k: [round(x, 3) for x in w] for k, w in rec.items()}, f"{time.time() - t0:.0f}s", flush=True)
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
