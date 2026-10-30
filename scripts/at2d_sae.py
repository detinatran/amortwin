"""Symbolic autoencoder: the twin's transport model is a fixed layer inside the network, not a target of the loss.

python scripts/at2d_sae.py train <cnn_dir> <x_dir> <folds.json> <work_dir> [--masks] [--epochs 60] [--rec-weight 1]
python scripts/at2d_sae.py infer <cnn_dir> <x_dir> - <work_dir> --masks --external pred

AmorTwin-2D supervises a U-Net with the twin's fitted parameters: the physics lives in the loss, and a network
trained without that loss keeps nothing of it. Here the same encoder instead emits the physical parameters of
every pixel, and a *fixed* symbolic decoder -- the indicator-dilution model of the twin, differentiable and
without a single learned weight -- rebuilds the observed part of the curve from them:

    encoder  : slice cut at L (4F channels)  ->  theta(v) = (ln MTT, delay, ln CBV) and log-variances
    symbolic : c_hat(v, t) = A exp(l_cbv - l_mtt) sum_{s<=t} a(s - delay) exp(-(t - s) / (M0 exp(l_mtt)))
    loss     : floored nMSE of (c_hat - c) over the frames actually measured  [+ the usual mask cross-entropy]

The reconstruction term needs no twin fit and no vendor mask: it is self-supervised on the scan itself, so it
also applies to a new centre. The twin targets remain available as an optional extra supervision (--twin-weight).

Known risk, stated before running (Sec. III-B of the paper): a scan cut at 25--35 s does not determine the MTT
(Cramer-Rao bound x8.7 at 30 s in hypoperfused tissue), so the reconstruction term constrains the delay far
better than the transit time. --detach-mtt therefore lets the reconstruction gradient reach the delay and the
amplitude only, leaving MTT to the mask head and the optional twin loss.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amortwin2d import inputs, load  # noqa: E402
from at2d_tta import M0_GRID, SMOOTH, rebuild, shifted_aif  # noqa: E402


def rebuild_beta(aif, q, L, m0):
    """rebuild() with the twin's dispersion: the arterial input is first smeared by the unit-area kernel
    exp(-t / beta) / beta (transport.disperse_rows), so Tmax = delay + t*(beta, MTT) of (2) is in the graph.
    q (P, 4) = (ln MTT, delay, ln CBV, ln beta)."""
    import torch
    a, ok = shifted_aif(aif, q[:, 1], L)
    ad = torch.exp(-1.0 / torch.exp(q[:, 3]).clamp(0.05, 30.0))       # one frame = 1 s
    base, u = a[:, :1], a - a[:, :1]
    ys = [torch.zeros_like(u[:, 0])]
    for n in range(1, L):
        ys.append(ad * ys[-1] + (1.0 - ad) * u[:, n])
    a = torch.stack(ys, 1) + base
    decay = torch.exp(-1.0 / (m0 * torch.exp(q[:, 0])))
    y, out = torch.zeros_like(q[:, 0]), []
    for k in range(L):
        y = y * decay + a[:, k]
        out.append(y)
    return torch.exp(q[:, 2] - q[:, 0])[:, None] * torch.stack(out, 1), ok


def gauss_kernel(dev, sigma=SMOOTH):
    """1-D Gaussian of the twin's width, as a conv kernel (the CPU round trip of at2d_tta.smooth_cut is the
    bottleneck when it runs once per slice of every batch)."""
    import torch
    r = int(4 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return (k / k.sum())[None, None, :], r


def smooth_gpu(c, k, r):
    """c (P, L) smoothed along time with edge padding, as scipy's mode='nearest'."""
    import torch
    x = torch.nn.functional.pad(c[:, None, :], (r, r), mode="replicate")
    return torch.nn.functional.conv1d(x, k)[:, 0, :]
from cnn_baseline import LENGTHS, net  # noqa: E402


def peak_time(y, ok):
    """Soft argmax over time of each rebuilt curve (P, L) -> (P,), in frames. A hard argmax has no gradient,
    so the peak is taken as a temperature-sharpened expectation; frames past the cut are excluded."""
    import torch
    w = torch.softmax(torch.where(ok, y, torch.full_like(y, -1e4)) * 8.0, dim=1)
    return (w * torch.arange(y.shape[1], dtype=y.dtype, device=y.device)[None, :]).sum(1)


def recon_loss(theta, c, aif, L, logA, logM, detach_mtt=False, beta=None, tmax=None, d0=None):
    """Floored nMSE between the symbolic reconstruction of theta (P, 3) and the observed curves c (P, L).

    Thin slabs (ISLES'18) hold pixels whose curve carries almost no signal, and a floor relative to the mean
    energy alone lets their ratio explode, so the loss is also clipped per pixel and the parameters are bounded.
    """
    import torch
    q = theta.clamp(-6, 6)               # ln MTT, delay and ln CBV cannot leave a physical range
    if d0 is not None:
        # the arrival offset of this patient's AIF is a nuisance, like the amplitude A and the transit scale
        # M0: with it, the delay head only carries the RELATIVE spatial pattern -- the quantity the vendor
        # hypoperfusion (relative Tmax) is defined on -- instead of an absolute timing that is centre-specific.
        q = torch.stack([q[:, 0], q[:, 1] + d0.clamp(-8.0, 8.0), q[:, 2]], 1)
    if detach_mtt:                       # the cut curve cannot identify the transit time (Sec. III-B)
        q = torch.stack([q[:, 0].detach(), q[:, 1], q[:, 2]], 1)
    m0 = torch.exp(logM.clamp(np.log(1.5), np.log(30.0)))
    if beta is None:
        y, ok = rebuild(aif, q, L, m0)
    else:                                 # the dispersion that enters Tmax joins the reconstruction
        y, ok = rebuild_beta(aif, torch.cat([q, beta.clamp(-3, 3.4)[:, None]], 1), L, m0)
    e = (c ** 2).sum(1)
    den = torch.clamp(e, min=max(0.05 * float(e.mean()), 1e-3))
    r = (ok * (torch.exp(logA.clamp(-6, 6)) * y - c) ** 2).sum(1) / den
    out = torch.nan_to_num(r, nan=0.0, posinf=0.0).clamp(max=50.0).mean()
    if tmax is not None:
        # Tmax is the peak time of the residue after the delay, which the rebuilt curve already carries:
        # tie the fourth head to it, so the symbolic decoder teaches the hypoperfusion as well as the core.
        tp = peak_time(y, ok)
        out = out + torch.nan_to_num((tmax.clamp(-5, 40) - tp) ** 2, nan=0.0, posinf=0.0).clamp(max=100.0).mean() / 25.0
    return out


def fit_scalars(theta, c, pix, aif, L):
    """Per-patient amplitude A and median transit scale M0 with the network frozen (grid over M0, closed form A)."""
    import torch
    best = None
    for m0 in M0_GRID:
        y, ok = rebuild(aif, theta, L, float(m0))
        A = float((ok * y * c).sum() / (ok * y * y).sum().clamp_min(1e-12))
        e = (c ** 2).sum(1)
        den = torch.clamp(e, min=0.05 * float(e.mean()))
        lo = float(((ok * (A * y - c) ** 2).sum(1) / den).mean())
        if best is None or lo < best[0]:
            best = (lo, A, float(m0))
    return best[1], best[2]


def train(a):
    import torch
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load(a.cnn, a.x)
    folds = {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    F = next(iter(data.values()))["ctc"].shape[1]
    dev = "cuda"
    a.work.mkdir(parents=True, exist_ok=True)
    for k in ([a.fold] if a.fold is not None else range(5)):
        wf = a.work / f"at2d_fold{k}.pt"
        if wf.exists():
            continue
        tr = [n for n in data if folds.get(n, -1) != k]
        wk = "trel" if a.reliable else "tmask"
        t_all = np.concatenate([data[n]["tgt"].astype(np.float32)[:, data[n][wk].astype(np.float32) > 0.5] for n in tr], 1)
        mu, sd = t_all.mean(1), t_all.std(1)
        np.savez(a.work / f"at2d_fold{k}_norm.npz", mu=mu, sd=sd)
        slices = [(n, z) for n in tr for z in range(data[n]["ctc"].shape[0])
                  if data[n]["tmask"][z].astype(np.float32).sum() > 50]
        model = net(F, cin=4 * F, cout=8 + (2 if a.masks else 0)).to(dev)
        # one amplitude and one transit scale per patient, learned jointly with the network
        pats = sorted({n for n, _ in slices})
        pi = {n: i for i, n in enumerate(pats)}
        logA = torch.zeros(len(pats), device=dev, requires_grad=True)
        logM = torch.full((len(pats),), float(np.log(6.0)), device=dev, requires_grad=True)
        d0 = torch.zeros(len(pats), device=dev, requires_grad=True)
        opt = torch.optim.AdamW(list(model.parameters()) + [logA, logM, d0], lr=1e-3, weight_decay=1e-4)
        steps = a.epochs * (len(slices) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        mu_t, sd_t = (torch.as_tensor(v, dtype=torch.float32, device=dev)[None, :, None, None] for v in (mu, sd))
        gk, gr = gauss_kernel(dev)
        t0 = time.time()
        for ep in range(a.epochs):
            order = rng.permutation(len(slices))
            for b in range(len(slices) // a.bs):
                X, T, W, Y, V, C, AIF, LS, PI = [], [], [], [], [], [], [], [], []
                for i in order[b * a.bs:(b + 1) * a.bs]:
                    n, z = slices[i]
                    d = data[n]
                    L = d["frames"] if rng.random() < 0.2 else int(rng.integers(25, d["frames"] + 1))
                    L = min(L, F)
                    x = inputs(d, L, [z])[0]
                    t = d["tgt"][:, z].astype(np.float32)
                    w = d[wk][z].astype(np.float32)
                    v = d["valid"][z].astype(np.float32)
                    y = np.stack([d["core"][z], d["hypo"][z]]).astype(np.float32) / np.maximum(v, 1e-3)
                    c = d["ctc"][z].astype(np.float32)
                    if rng.random() < 0.5:        # the curves must follow the flip, or the decoder rebuilds
                        x, t, w, y, v, c = (u[..., ::-1] for u in (x, t, w, y, v, c))   # the wrong hemisphere
                    X.append(x), T.append(t), W.append(w), Y.append(np.clip(y, 0, 1)), V.append(np.stack([v, v]))
                    C.append(np.ascontiguousarray(c)), AIF.append(d["aif"]), LS.append(L), PI.append(pi[n])
                X, T, W, Y, V = (torch.as_tensor(np.ascontiguousarray(np.stack(u))).to(dev) for u in (X, T, W, Y, V))
                out = model(X)
                m, lv = out[:, :4], out[:, 4:8].clamp(-8, 6)
                loss = 0.0
                if a.twin_weight:
                    z_ = (T - mu_t) / sd_t
                    nll = 0.5 * (lv + (z_ - m) ** 2 * torch.exp(-lv))
                    loss = loss + a.twin_weight * (nll * W[:, None]).sum() / W.sum().clamp_min(1) / 4
                if a.masks:
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 8:10], Y, reduction="none")
                    loss = loss + (bce * V).sum() / V.sum().clamp_min(1)
                if a.rec_weight:                 # symbolic decoder on the observed part of each slice
                    rec, nrec = 0.0, 0
                    for j in range(X.shape[0]):
                        L = LS[j]
                        valid = V[j, 0] > 0.5        # already on the device; no host round trip
                        if valid.sum() < 50:
                            continue
                        th = (m[j] * sd_t[0, :, 0, 0][:, None, None] + mu_t[0, :, 0, 0][:, None, None])
                        th = th[:3].permute(1, 2, 0)[valid]              # (P, 3) = ln MTT, delay, ln CBV
                        cc = smooth_gpu(torch.as_tensor(C[j], device=dev).permute(1, 2, 0)[valid][:, :L], gk, gr)
                        aif = torch.as_tensor(AIF[j], dtype=torch.float32, device=dev).clone()
                        aif[L:] = 0
                        bq = tq = None
                        if a.with_beta:   # the fourth head output is read as ln beta
                            bq = (m[j, 3] * sd_t[0, 3, 0, 0] + mu_t[0, 3, 0, 0])[valid]
                        if a.peak_tmax:   # the fourth head is Tmax, tied to the rebuilt curve's peak time
                            tq = (m[j, 3] * sd_t[0, 3, 0, 0] + mu_t[0, 3, 0, 0])[valid]
                        rec = rec + recon_loss(th, cc, aif, L, logA[PI[j]], logM[PI[j]], a.detach_mtt, bq, tq,
                                               d0[PI[j]] if a.delay_offset else None)
                        nrec += 1
                    if nrec:
                        loss = loss + a.rec_weight * rec / nrec
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [logA, logM, d0], 5.0)
                opt.step()
                sched.step()
            print(f"fold {k} epoch {ep + 1}/{a.epochs} loss {float(loss):.4f} "
                  f"M0 {float(torch.exp(logM).median()):.1f} D0 {float(d0.median()):+.1f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), wf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("train", "infer", "eval"))
    p.add_argument("args", nargs="+")
    p.add_argument("--masks", action="store_true")
    p.add_argument("--rec-weight", type=float, default=1.0, help="weight of the symbolic reconstruction term")
    p.add_argument("--twin-weight", type=float, default=0.0, help="weight of the twin-target loss (0: none)")
    p.add_argument("--delay-offset", action="store_true",
                   help="a per-patient arrival offset joins A and M0, so the delay head is relative")
    p.add_argument("--peak-tmax", action="store_true",
                   help="tie the Tmax head to the peak time of the rebuilt curve (teaches the hypoperfusion)")
    p.add_argument("--with-beta", action="store_true",
                   help="the symbolic decoder also disperses the input, so Tmax enters the reconstruction")
    p.add_argument("--detach-mtt", action="store_true", help="reconstruction gradient skips ln MTT (Sec. III-B)")
    p.add_argument("--reliable", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--fold", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--external")
    p.add_argument("--ext-refs")
    p.add_argument("--level", type=float, default=0.8)
    p.add_argument("--shift", type=int, default=0)
    p.add_argument("--swap")
    p.add_argument("--swap-from", default="at2d24x")
    p.add_argument("--phys-weight", type=float, default=1.0)
    p.add_argument("--adapt-cnn")
    p.add_argument("--adapt-x")
    p.add_argument("--adapt-folds")
    p.add_argument("--adapt-fold", type=int, default=0)
    p.add_argument("--pseudo")
    a = p.parse_args()
    if a.stage == "train":
        a.cnn, a.x, a.folds, a.work = a.args[0], a.args[1], a.args[2], Path(a.args[3])
        train(a)
        return
    from amortwin2d import evaluate, infer           # inference and evaluation are unchanged
    if a.stage == "eval":
        a.work, a.refs, a.out = Path(a.args[0]), a.args[1], a.args[2]
        evaluate(a)
        return
    a.cnn, a.x, a.folds, a.work = a.args[0], a.args[1], a.args[2], Path(a.args[3])
    infer(a)


if __name__ == "__main__":
    main()
