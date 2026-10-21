"""AmorTwin training, for a shared GPU.

python amortized_train.py <pairs.npz> <out_dir> [--epochs 30]
python amortized_train.py <pairs with the Tmax target> <out_dir> --trunc-min 25 --prob --augment

A 1D network maps a voxel curve + AIF (+ the patient's median tissue curve) to the twin voxel
parameters ln(MTT / brain median), delay, ln(CBV / brain median). Five models, one per patient
fold of the duofusion evaluation (model k never sees fold k). The raw arrays stay compact on the
GPU (curves in fp16, AIF / context per patient) and the network inputs are built per batch, so
the run needs ~1-2 GB of GPU memory next to nnU-Net. A fold whose weights exist is skipped.
--trunc-min L0 (E3): every training curve is cut at a random length in [L0 s, its full length]
(with --short-frac f, a share f of the cuts is drawn in [L0, --short-max] s instead)
(the frame mask of make_inputs hides the rest of the curve, the AIF and the context), while the
targets stay those of the full scan; the held-out metrics are then reported for the full scan and
at 25, 30 and 35 s. The number of outputs follows the targets of the pairs (3, or 4 with Tmax).
--prob: probabilistic AmorTwin, the head gives a mean and a log-variance per target, trained by the
Gaussian negative log-likelihood (on the standardised targets); the held-out metrics then add the
coverage of the 80 % and 90 % central intervals, the mean predicted SD, and the Spearman correlation
between the predicted SD and the absolute error (does the network know when it does not know?).
--augment: acquisition augmentation of the training inputs, per batch: Gaussian noise of U[0, 15 %] of
each curve's peak on the voxel curve; with probability 0.5 a later start (the first 0-6 frames dropped
from the curve, the median tissue curve and the AIF, shorter length); with probability 0.3 a coarser
sampling (every 2nd or 3rd frame, linear interpolation back to 1 s). Targets are unchanged.
--compose-weight w (pairs with the TARGETS6 targets and --prob): adds w x the Gaussian NLL of the twin
Tmax under the Tmax composed from the network's own delay, ln MTT and ln dispersion posteriors
(Tmax = delay + residue peak(beta, MTT), reparameterised Monte Carlo, 8 draws), so that the physics
composition is trained end to end instead of only applied after training.
Writes amortized_fold<k>.pt and heldout_metrics.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from amortized_model import IN_CH, IN_CH_ROI, TARGETS, TARGETS4, TARGETS6, AmorTwin, composed_tmax, make_inputs, residue_peak_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pairs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--trunc-min", type=int, default=0, help="E3: random truncation of the inputs from this length (s)")
    p.add_argument("--short-frac", type=float, default=0.0, help="share of samples cut in [trunc-min, short-max] s")
    p.add_argument("--short-max", type=int, default=35)
    p.add_argument("--prob", action="store_true", help="Gaussian head (mean, log-variance), NLL loss")
    p.add_argument("--augment", action="store_true", help="noise / late start / coarse sampling augmentation")
    p.add_argument("--compose-weight", type=float, default=0.0, help="TARGETS6 + --prob: weight of the composed-Tmax NLL")
    p.add_argument("--train-frac", type=float, default=1.0, help="learning curve: share of the training patients of each fold")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    z = np.load(a.pairs)
    D = {k: z[k] for k in z.files}
    print({k: v.shape for k, v in D.items()}, flush=True)
    P = torch.tensor(D["patient_index"].astype(np.int64), device=dev)
    CUR = torch.tensor(D["curve"], dtype=torch.float16, device=dev)
    CTX = torch.tensor(D["ctx"], dtype=torch.float32, device=dev)
    AIF = torch.tensor(D["aif"], dtype=torch.float32, device=dev)
    LEN = torch.tensor(D["length"].astype(np.int64), device=dev)
    TIS = torch.tensor(D["tissue"].astype(np.int64), device=dev)
    Y = torch.tensor(D["target"], dtype=torch.float32, device=dev)
    FOLD = torch.tensor(D["fold"].astype(np.int64), device=dev)
    ROIC = "roi_curve" in D             # pairs of export_pairs.py --roi-context: the ROI median curve as input
    if ROIC:
        RC = torch.tensor(D["roi_curve"], dtype=torch.float32, device=dev)
        RI = torch.tensor(D["roi_index"].astype(np.int64), device=dev)
    names = {3: TARGETS, 4: TARGETS4, 6: TARGETS6}[Y.shape[1]]
    mu, sd = Y.mean(0), Y.std(0)
    print("voxels", len(Y), "target mean", mu.cpu().numpy(), "sd", sd.cpu().numpy(), flush=True)

    def inputs(i, length=None):
        return make_inputs(CUR[i].float(), CTX[P[i]], AIF[P[i]], LEN[i] if length is None else length, TIS[i],
                           RC[RI[i]] if ROIC else None)

    def resample(x, step):
        T_ = x.shape[1]
        keep = torch.arange(0, T_, step, device=x.device)
        t = torch.arange(T_, device=x.device, dtype=torch.float32)
        idx = torch.clamp(torch.searchsorted(keep.float(), t, right=True) - 1, 0, len(keep) - 2)
        t0, t1 = keep[idx].float(), keep[idx + 1].float()
        w = ((t - t0) / (t1 - t0)).clamp(0, 1)
        return x[:, keep[idx]] * (1 - w) + x[:, keep[idx + 1]] * w

    def augmented_inputs(i, length):
        cur, ctx, aif = CUR[i].float(), CTX[P[i]], AIF[P[i]]
        roi = RC[RI[i]] if ROIC else torch.zeros_like(ctx)
        ln = LEN[i] if length is None else length
        if torch.rand(1).item() < 0.5:
            k = int(torch.randint(0, 7, (1,)).item())
            if k:
                drop = lambda x: torch.cat([x[:, k:], torch.zeros_like(x[:, :k])], 1)  # noqa: E731
                cur, ctx, aif, roi, ln = drop(cur), drop(ctx), drop(aif), drop(roi), torch.clamp(ln - k, min=5)
        if torch.rand(1).item() < 0.3:
            st = int(torch.randint(2, 4, (1,)).item())
            cur, ctx, aif, roi = resample(cur, st), resample(ctx, st), resample(aif, st), resample(roi, st)
        lvl = 0.15 * torch.rand(len(i), 1, device=dev)
        cur = cur + lvl * cur.amax(1, keepdim=True).clamp(min=1e-3) * torch.randn_like(cur)
        return make_inputs(cur, ctx, aif, ln, TIS[i], roi if ROIC else None)

    def random_length(i):
        full = LEN[i]
        lo = torch.clamp(torch.full_like(full, a.trunc_min), max=full)
        L = lo + (torch.rand(len(i), device=dev) * (full - lo + 1).float()).long().clamp(max=full - lo)
        if a.short_frac > 0:
            hi = torch.clamp(torch.full_like(full, a.short_max), min=lo, max=full)
            Ls = lo + (torch.rand(len(i), device=dev) * (hi - lo + 1).float()).long().clamp(max=hi - lo)
            L = torch.where(torch.rand(len(i), device=dev) < a.short_frac, Ls, L)
        return L

    PIDS = D["pid"].astype(np.int64)

    def train_fold(k):
        trn = FOLD.cpu().numpy() != k
        if a.train_frac < 1.0:
            pats = np.unique(PIDS[trn])
            keep = np.random.default_rng(1000 + k).choice(pats, max(1, int(round(a.train_frac * len(pats)))), replace=False)
            trn &= np.isin(PIDS, keep)
        tr = torch.as_tensor(np.flatnonzero(trn), device=dev)
        K = len(names)
        model = AmorTwin(n_out=2 * K if a.prob else K, in_ch=IN_CH_ROI if ROIC else IN_CH).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        steps = a.epochs * (len(torch.nonzero(FOLD != k)) // a.bs)      # same number of steps for every train-frac
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
        model.train()
        for s in range(steps):
            i = tr[torch.randint(0, len(tr), (a.bs,), device=dev)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ln_ = random_length(i) if a.trunc_min else None
                pred = model(augmented_inputs(i, ln_) if a.augment else inputs(i, ln_))
            yt = (Y[i] - mu) / sd
            if a.prob:
                m_, lv = pred.float()[:, :K], pred.float()[:, K:].clamp(-9.0, 6.0)
                loss = 0.5 * (lv + (yt - m_) ** 2 * torch.exp(-lv)).mean()
                if a.compose_weight > 0:
                    tm, ts = composed_tmax(m_ * sd + mu, torch.exp(0.5 * lv) * sd, n=8)
                    vc = (ts / sd[3]) ** 2 + 1e-3
                    loss = loss + a.compose_weight * 0.5 * (torch.log(vc) + ((Y[i, 3] - tm) / sd[3]) ** 2 / vc).mean()
            else:
                loss = F.smooth_l1_loss(pred.float(), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if s % 1000 == 0:
                print(f"fold {k} step {s}/{steps} loss {loss.item():.4f}", flush=True)
        return model

    @torch.no_grad()
    def evaluate(model, k):
        te = torch.nonzero(FOLD == k).squeeze(1)
        model.eval()
        y = Y[te]
        out = {}
        for L in (["full", 25, 30, 35] if a.trunc_min else ["full"]):
            def length(idx):
                return None if L == "full" else torch.clamp(LEN[idx], max=int(L))
            raw = torch.cat([model(inputs(te[s:s + 65536], length(te[s:s + 65536]))).float()
                             for s in range(0, len(te), 65536)])
            K = len(names)
            pred = raw[:, :K] * sd + mu
            psd = torch.exp(0.5 * raw[:, K:].clamp(-9.0, 6.0)) * sd if a.prob else None
            res = {}
            for j, name in enumerate(names):
                e = (pred[:, j] - y[:, j]).abs()
                r = np.corrcoef(pred[:, j].cpu().numpy(), y[:, j].cpu().numpy())[0, 1]
                res[name] = {"mae": float(e.mean()), "median_ae": float(e.median()), "r": float(r)}
                if a.prob:
                    z = e / psd[:, j]
                    sub = torch.randperm(len(e), device=dev)[:200000]
                    from scipy.stats import spearmanr
                    rho = spearmanr(psd[sub, j].cpu().numpy(), e[sub].cpu().numpy()).correlation
                    res[name].update({"cov80": float((z <= 1.2816).float().mean()), "cov90": float((z <= 1.6449).float().mean()),
                                      "mean_sd": float(psd[:, j].mean()), "sd_err_spearman": float(rho)})
            if len(names) == 6:                       # Tmax composed from delay, MTT and dispersion
                if a.prob:
                    tm, _ = composed_tmax(pred, psd)
                else:
                    tm = pred[:, 1] + residue_peak_t(torch.exp(pred[:, 5]), torch.exp(pred[:, 4]))
                e = (tm - y[:, 3]).abs()
                hy = y[:, 3] > 6
                res["tmax_composed"] = {"mae": float(e.mean()), "r": float(np.corrcoef(tm.cpu().numpy(), y[:, 3].cpu().numpy())[0, 1]),
                                        "mae_tmax_gt6": float(e[hy].mean()), "bias_tmax_gt6": float((tm - y[:, 3])[hy].mean()),
                                        "mae_head_tmax_gt6": float((pred[:, 3] - y[:, 3]).abs()[hy].mean())}
            out[str(L)] = res
        return out if a.trunc_min else out["full"]

    mfile = a.out / "heldout_metrics.json"
    metrics = json.loads(mfile.read_text()) if mfile.exists() else {}
    for k in range(5):
        w = a.out / f"amortized_fold{k}.pt"
        if w.exists() and str(k) in metrics:
            continue
        t0 = time.time()
        model = train_fold(k)
        torch.save({"state": model.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "targets": list(names), "compose_weight": a.compose_weight, "roi_context": ROIC,
                    "trunc_min": a.trunc_min, "short_frac": a.short_frac, "short_max": a.short_max}, w)
        metrics[str(k)] = evaluate(model, k)
        mfile.write_text(json.dumps(metrics, indent=1))
        print("fold", k, f"{time.time() - t0:.0f}s", json.dumps(metrics[str(k)]), flush=True)
    print("AMORTIZED_DONE", flush=True)


if __name__ == "__main__":
    main()
