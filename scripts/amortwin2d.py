"""Spatial AmorTwin (AmorTwin-2D): the amortized twin posterior with the context of a U-Net.

python scripts/amortwin2d.py train <cnn24_dir> <at2d24x_dir> <folds.json> <work_dir> [--masks] [--epochs 60]
python scripts/amortwin2d.py infer <cnn24_dir> <at2d24x_dir> <folds.json> <work_dir> [--masks]
python scripts/amortwin2d.py eval  <work_dir> <refs24.json> <out.json>
python scripts/amortwin2d.py infer <cnn18_dir> <at2d18x_dir> - <work_dir> [--masks] --external pred18
python scripts/amortwin2d.py eval  <work_dir> <refs24.json> <out18.json> --external pred18 --ext-refs <refs18.json>

AmorTwin returns a Gaussian posterior of the twin's full-scan quantities (ln MTT/median, delay, ln CBV/median,
Tmax) from one voxel curve cut at L. AmorTwin-2D returns the same posterior for every pixel of a slice from the
whole cut slice, as the end-to-end CNN of cnn_baseline.py does (same U-Net), so that a voxel sees its
neighbourhood. Inputs (4F channels): the cut curves, the frame mask, and the cut arterial input and median
tissue curve broadcast over the slice. Loss: Gaussian NLL of the standardised targets, weighted by the share of
twin-fitted tissue in the pixel. Training lengths as AmorTwin and the CNN: L ~ U[25 s, length], full with p 0.2.
--masks adds the CNN's two vendor-mask outputs (multi-task); without it no vendor label is used in training.
--phys-weight 0 with --masks is the control: the same inputs and network trained on the masks only.
--adapt-cnn/--adapt-x/--adapt-folds/--adapt-fold k: label-free adaptation to another centre. Every step adds a batch of
the same size drawn from the full scans of the other centre's patients outside test half k (twin targets only, no
mask loss), cut at random lengths like the source slices; the source batches and the number of steps are unchanged.
The twin targets need no label, only a twin fit of past full scans of the new centre. The test half k is never seen.
--pseudo <work_dir> (with --adapt-*, for the control): self-training baseline. The target slices get hard pseudo-masks
from the mask outputs of the source-trained model at2d_fold0.pt of <work_dir> on the full scan (teacher), counted only where
the teacher is confident (p <= 0.1 or >= 0.9, per mask); the student is trained from scratch on them, cut at random lengths.
--reliable: the twin loss (and the standardisation of the targets) only counts pixels whose twin fit has a regional
dispersion beta <= 1 s, the fits whose Tmax is identifiable (Sec. III-C of the paper); the volumes still use tmask.

Volumes, as the AmorTwin pipeline: core = rCBF (exp(ln CBV - ln MTT), relative to the median of the less delayed
hemisphere) < tau_c; hypoperfusion = relative Tmax (Tmax minus the median of the less delayed hemisphere,
smoothed with sigma 1.5 pixels, about 3 mm) > d; both weighted by the valid-tissue share. tau_c and d are chosen on
the full-scan points of the training folds to match the vendor volumes, and applied unchanged to the cut scans.
The posterior gives hypoperfusion volumes at 11 comonotone quantiles of Tmax (5 to 95 %); the 10/90 % volumes
form the raw interval, calibrated by normalised conformal prediction as in prob_hybrid.py. eval reports the same
quantities as cnn_baseline.py eval; with --masks also the mask outputs (CNN-style volumes).
--external (another cohort, no retraining): every patient through the five fold models, mean of the means and
total variance; eval then chooses the thresholds on the full scans of all ISLES'24 patients and takes the
conformal quantile of all ISLES'24 scores, and evaluates the other cohort. The per-patient points are saved
in the output for the comparison with the follow-up infarct.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_baseline import LENGTHS, net  # noqa: E402

QS = np.linspace(0.05, 0.95, 11)
TAUS = np.round(np.arange(0.02, 0.6, 0.02), 3)
DS = np.round(np.arange(-2.0, 10.01, 0.25), 2)


def load(cnn, x):
    data = {}
    for f in sorted(Path(x).glob("training_*.npz")):
        n = int(f.stem.split("_")[1])
        z, c = np.load(f), np.load(Path(cnn) / f.name)
        data[n] = {k: c[k] for k in ("ctc", "valid", "core", "hypo")} | {k: z[k] for k in ("tgt", "tmask", "hemi", "aif", "ctx")} | {
            "trel": z["trel"] if "trel" in z.files else z["tmask"]} | {
            "frames": int(c["frames"]), "pixel_ml": float(c["pixel_ml"])}
    return data


def shift_start(d, k):
    """The scan starting k frames later: the first k frames of the curves, AIF and tissue curve dropped."""
    if k <= 0:
        return d
    F = d["ctc"].shape[1]
    pad = lambda x, ax: np.concatenate([np.take(x, np.arange(k, F), axis=ax), np.zeros_like(np.take(x, np.arange(k), axis=ax))], ax)  # noqa: E731
    return d | {"ctc": pad(d["ctc"], 1), "aif": pad(d["aif"], 0), "ctx": pad(d["ctx"], 0), "frames": d["frames"] - k}


def inputs(d, L, zs=None):
    """(Z or len(zs), 4F, S, S): cut curves, frame mask, cut AIF and median tissue curve."""
    ctc = d["ctc"] if zs is None else d["ctc"][zs]
    ctc = ctc.astype(np.float32)
    Z, F, S, _ = ctc.shape
    m = (np.arange(F) < L).astype(np.float32)
    bc = lambda v: np.broadcast_to((v * m)[None, :, None, None], (Z, F, S, S))  # noqa: E731
    return np.concatenate([ctc * m[None, :, None, None], bc(np.ones(F, np.float32)), bc(d["aif"]), bc(d["ctx"])], 1)


def train(a):
    import torch
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load(a.cnn, a.x)
    tdata = load(a.adapt_cnn, a.adapt_x) if a.adapt_cnn else {}
    folds = {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    F = next(iter(data.values()))["ctc"].shape[1]
    dev = "cuda"
    a.work.mkdir(parents=True, exist_ok=True)
    for k in ([a.fold] if a.fold is not None else range(5)):
        wf = a.work / f"at2d_fold{k}.pt"
        if wf.exists():
            continue
        tr = [n for n in data if folds.get(n, -1) != k]
        wk = "trel" if a.reliable else "tmask"       # --reliable: twin loss only where the dispersion is identifiable
        t_all = np.concatenate([data[n]["tgt"].astype(np.float32)[:, data[n][wk].astype(np.float32) > 0.5] for n in tr], 1)
        mu, sd = t_all.mean(1), t_all.std(1)
        np.savez(a.work / f"at2d_fold{k}_norm.npz", mu=mu, sd=sd)
        slices = [(n, z) for n in tr for z in range(data[n]["ctc"].shape[0]) if data[n]["tmask"][z].astype(np.float32).sum() > 50]
        tslices = []
        if a.adapt_cnn:
            tf = {int(k_): v for k_, v in json.loads(Path(a.adapt_folds).read_text()).items()}
            tslices = [(n, z) for n in tdata if tf.get(n, -1) != a.adapt_fold for z in range(tdata[n]["ctc"].shape[0])
                       if tdata[n]["tmask"][z].astype(np.float32).sum() > 50]
            print(f"adaptation: {len(tslices)} slices of {len({n for n, _ in tslices})} target patients (test half {a.adapt_fold} held out)", flush=True)
        pseudo = {}
        if a.pseudo:         # teacher mask probabilities on the full scans of the adaptation patients
            teacher = net(F, cin=4 * F, cout=10).to(dev).eval()
            teacher.load_state_dict(torch.load(Path(a.pseudo) / "at2d_fold0.pt", map_location=dev))
            for n in sorted({n for n, _ in tslices}):
                with torch.no_grad():
                    pseudo[n] = torch.sigmoid(teacher(torch.as_tensor(inputs(tdata[n], tdata[n]["frames"])).to(dev))[:, 8:10]).cpu().numpy()
            del teacher
            conf = np.mean([np.mean(np.abs(pseudo[n][z] - 0.5) >= 0.4) for n, z in tslices])
            print(f"pseudo-masks from {a.pseudo}: confident share {conf:.3f}", flush=True)
        model = net(F, cin=4 * F, cout=8 + (2 if a.masks else 0)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = a.epochs * (len(slices) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        mu_t, sd_t = (torch.as_tensor(v, dtype=torch.float32, device=dev)[None, :, None, None] for v in (mu, sd))
        t0 = time.time()
        for ep in range(a.epochs):
            order = rng.permutation(len(slices))
            for b in range(len(slices) // a.bs):
                X, T, W, Y, V = [], [], [], [], []
                batch = [(data, slices[i], False) for i in order[b * a.bs:(b + 1) * a.bs]]
                if tslices:
                    batch += [(tdata, tslices[i], True) for i in rng.integers(0, len(tslices), a.bs)]
                for src, (n, z), is_target in batch:
                    d = src[n]
                    L = d["frames"] if rng.random() < 0.2 else int(rng.integers(25, d["frames"] + 1))
                    x = inputs(d, L, [z])[0]
                    t = d["tgt"][:, z].astype(np.float32)
                    w = d[wk][z].astype(np.float32)
                    v = d["valid"][z].astype(np.float32)
                    y = np.stack([d["core"][z], d["hypo"][z]]).astype(np.float32) / np.maximum(v, 1e-3)
                    v = np.stack([v, v])
                    if is_target and a.pseudo:   # confident teacher pseudo-masks instead of the labels
                        p = pseudo[n][z]
                        y, v = (p > 0.5).astype(np.float32), v * (np.abs(p - 0.5) >= 0.4)
                    elif is_target:        # no label on the new centre: twin loss only
                        v = np.zeros_like(v)
                    if rng.random() < 0.5:
                        x, t, w, y, v = x[..., ::-1], t[..., ::-1], w[..., ::-1], y[..., ::-1], v[..., ::-1]
                    X.append(x), T.append(t), W.append(w), Y.append(np.clip(y, 0, 1)), V.append(v)
                X, T, W, Y, V = (torch.as_tensor(np.ascontiguousarray(np.stack(u))).to(dev) for u in (X, T, W, Y, V))
                out = model(X)
                m, lv = out[:, :4], out[:, 4:8].clamp(-8, 6)
                z_ = (T - mu_t) / sd_t
                nll = 0.5 * (lv + (z_ - m) ** 2 * torch.exp(-lv))
                loss = a.phys_weight * (nll * W[:, None]).sum() / W.sum().clamp_min(1) / 4
                if a.masks:
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 8:10], Y, reduction="none")
                    loss = loss + (bce * V).sum() / V.sum().clamp_min(1)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
            print(f"fold {k} epoch {ep + 1}/{a.epochs} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), wf)


def infer(a):
    import torch
    data = load(a.cnn, a.x)
    if a.shift:
        data = {n: shift_start(d, a.shift) for n, d in data.items()}
    if a.swap:      # diagnostic: replace the AIF and/or tissue curve by the mean ISLES'24 one aligned at its peak
        tmpl = {}
        for key in ("aif", "ctx"):
            curves = [np.load(f)[key] for f in sorted(Path(a.swap_from).glob("*.npz"))]
            al = [np.roll(c, 20 - int(np.argmax(c))) for c in curves]
            tmpl[key] = np.mean(al, 0)
        for d in data.values():
            for key in (("aif", "ctx") if a.swap == "both" else (a.swap,)):
                t = np.roll(tmpl[key], int(np.argmax(d[key])) - 20)
                d[key] = (t / t.max()).astype(np.float32)
    folds = {} if a.external else {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    F = next(iter(data.values()))["ctc"].shape[1]
    dev = "cuda"
    od = a.work / (a.external or "pred")
    od.mkdir(parents=True, exist_ok=True)
    if a.external:
        models = []
        for k in range(5):
            m = net(F, cin=4 * F, cout=8 + (2 if a.masks else 0)).to(dev).eval()
            m.load_state_dict(torch.load(a.work / f"at2d_fold{k}.pt", map_location=dev))
            models.append((m, np.load(a.work / f"at2d_fold{k}_norm.npz")))
        for n, d in data.items():
            rec = {}
            for Ls in LENGTHS:
                L = d["frames"] if Ls == "full" else int(Ls)
                mus, vs, ps = [], [], []
                for m, nz in models:
                    with torch.no_grad():
                        o = m(torch.as_tensor(inputs(d, L)).to(dev)).cpu().numpy()
                    mus.append(o[:, :4] * nz["sd"][None, :, None, None] + nz["mu"][None, :, None, None])
                    vs.append((np.exp(0.5 * np.clip(o[:, 4:8], -8, 6)) * nz["sd"][None, :, None, None]) ** 2)
                    if a.masks:
                        ps.append(1 / (1 + np.exp(-o[:, 8:10])))
                mus = np.stack(mus)
                rec[f"mu_{Ls}"] = mus.mean(0).astype(np.float16)
                rec[f"sd_{Ls}"] = np.sqrt(np.mean(vs, 0) + mus.var(0)).astype(np.float16)
                if a.masks:
                    rec[f"mask_{Ls}"] = np.mean(ps, 0).astype(np.float16)
            np.savez_compressed(od / f"training_{n}.npz", **rec, valid=d["valid"], tmask=d["tmask"], hemi=d["hemi"],
                                pixel_ml=d["pixel_ml"])
            print(n, "ok", flush=True)
        return
    for k in range(5):
        model = net(F, cin=4 * F, cout=8 + (2 if a.masks else 0)).to(dev).eval()
        model.load_state_dict(torch.load(a.work / f"at2d_fold{k}.pt", map_location=dev))
        nz = np.load(a.work / f"at2d_fold{k}_norm.npz")
        for n, d in data.items():
            if folds.get(n, -1) != k:
                continue
            rec = {}
            for Ls in LENGTHS:
                L = d["frames"] if Ls == "full" else int(Ls)
                with torch.no_grad():
                    o = model(torch.as_tensor(inputs(d, L)).to(dev)).cpu().numpy()
                mu = o[:, :4] * nz["sd"][None, :, None, None] + nz["mu"][None, :, None, None]
                sd = np.exp(0.5 * np.clip(o[:, 4:8], -8, 6)) * nz["sd"][None, :, None, None]
                rec[f"mu_{Ls}"], rec[f"sd_{Ls}"] = mu.astype(np.float16), sd.astype(np.float16)
                if a.masks:
                    rec[f"mask_{Ls}"] = (1 / (1 + np.exp(-o[:, 8:10]))).astype(np.float16)
            np.savez_compressed(od / f"training_{n}.npz", **rec, valid=d["valid"], tmask=d["tmask"], hemi=d["hemi"],
                                pixel_ml=d["pixel_ml"])
            print(n, "ok", flush=True)


def volumes(z, Ls):
    """Core volume per tau (TAUS), hypoperfusion volume per d (DS) and Tmax quantile (QS)."""
    mu, sd = z[f"mu_{Ls}"].astype(np.float32), z[f"sd_{Ls}"].astype(np.float32)
    v = z["valid"].astype(np.float32)
    ok = z["tmask"].astype(np.float32) > 0.5
    hemi = z["hemi"]
    tm = mu[:, 3]
    meds = [np.median(tm[ok & (hemi == h)]) for h in (1, 2) if np.any(ok & (hemi == h))]
    ref = int(np.argmin(meds)) + 1 if len(meds) == 2 else 1
    sel = ok & (hemi == ref) if np.any(ok & (hemi == ref)) else ok
    lcbf = mu[:, 2] - mu[:, 0]
    rcbf = np.exp(lcbf - np.median(lcbf[sel]))
    pm = float(z["pixel_ml"])
    core = np.array([float((v * (ok & (rcbf < t))).sum()) * pm for t in TAUS])
    base = np.median(tm[sel])

    def smooth(x):
        num = np.stack([gaussian_filter(np.where(ok[i], x[i], 0.0), 1.5) for i in range(x.shape[0])])
        den = np.stack([gaussian_filter(ok[i].astype(np.float32), 1.5) for i in range(x.shape[0])])
        return np.where(ok, num / np.maximum(den, 1e-6), -1e9)

    hyp = np.zeros((len(QS), len(DS)))
    for i, q in enumerate(QS):
        trel = smooth(tm + norm.ppf(q) * sd[:, 3] - base)
        for j, dd in enumerate(DS):
            hyp[i, j] = float((v * (trel > dd)).sum()) * pm
    out = {"core": core, "hypo": hyp}
    if f"mask_{Ls}" in z:
        p = z[f"mask_{Ls}"].astype(np.float32)
        out["mask"] = [float((p[:, 0] * v).sum()) * pm, float((p[:, 1] * v).sum()) * pm]
    return out


def cohort(pred_dir, refs_file):
    refs = json.loads(Path(refs_file).read_text())
    core_ref, hypo_ref, dec_ref = ({int(n): v for n, v in refs[k].items()} for k in ("core", "hypo", "dec"))
    groups = {int(n): g for n, g in refs["groups"].items()}
    files = {int(f.stem.split("_")[1]): f for f in Path(pred_dir).glob("training_*.npz")}
    pats = sorted(n for n in files if n in core_ref)
    V = {}
    for n in pats:
        z = np.load(files[n])
        V[n] = {Ls: volumes(z, Ls) for Ls in LENGTHS}
    return pats, V, core_ref, hypo_ref, dec_ref, groups


def evaluate(a):
    from truncation_accuracy import defuse3, folds as make_folds
    pats, V, core_ref, hypo_ref, dec_ref, groups = cohort(a.work / "pred", a.refs)
    fold = make_folds({n: groups[n] for n in pats})
    ext = None
    if a.external:       # calibration: all ISLES'24 patients (one pseudo-fold -1); evaluation: the other cohort
        ext = cohort(a.work / a.external, a.ext_refs)
        cal_pats, calV, cal_core, cal_hypo = pats, V, core_ref, hypo_ref
        pats, V, core_ref, hypo_ref, dec_ref, groups = ext
        fold = {n: -1 for n in pats}
    imid = len(QS) // 2
    jc, jd = {}, {}
    for k in set(fold.values()):
        tr = [n for n in pats if fold[n] != k]
        if ext:
            tr, Vt, cr, hr = cal_pats, calV, cal_core, cal_hypo
        else:
            Vt, cr, hr = V, core_ref, hypo_ref
        ec = [np.median([abs(np.log1p(Vt[n]["full"]["core"][i]) - np.log1p(cr[n])) for n in tr]) for i in range(len(TAUS))]
        eh = [np.median([abs(np.log1p(Vt[n]["full"]["hypo"][imid, j]) - np.log1p(hr[n])) for n in tr]) for j in range(len(DS))]
        jc[k], jd[k] = int(np.argmin(ec)), int(np.argmin(eh))
    print("thresholds per fold:", {k: (float(TAUS[jc[k]]), float(DS[jd[k]])) for k in jc})
    d3 = lambda c, h: defuse3(c, max(c, h))  # noqa: E731
    iq = [int(np.argmin(np.abs(QS - q))) for q in ((1 - a.level) / 2, (1 + a.level) / 2)]
    out = {"n": len(pats), "points": {}}
    variants = {"physics": lambda n, Ls, VV=None: ((VV or V)[n][Ls]["core"][jc[fold.get(n, -1)]],
                                                     (VV or V)[n][Ls]["hypo"][imid, jd[fold.get(n, -1)]])}
    if "mask" in V[pats[0]]["full"]:
        variants["masks"] = lambda n, Ls, VV=None: tuple((VV or V)[n][Ls]["mask"])
    for name, point in variants.items():
        out[name] = {}
        out["points"][name] = {Ls: {str(n): list(map(float, point(n, Ls))) for n in pats} for Ls in LENGTHS}
        for Ls in LENGTHS:
            pt = {n: point(n, Ls) for n in pats}
            full = {n: point(n, "full") for n in pats}
            res = {"core_mae_ml": float(np.median([abs(pt[n][0] - core_ref[n]) for n in pats])),
                   "hypo_mae_ml": float(np.median([abs(pt[n][1] - hypo_ref[n]) for n in pats]))}
            flip = np.array([d3(*pt[n]) != d3(*full[n]) for n in pats])
            agree = np.array([d3(*pt[n]) == dec_ref[n] for n in pats])
            res["flips"], res["agree"] = float(flip.mean()), float(agree.mean())
            if Ls != "full":
                for ref_kind in ("self", "vendor"):
                    for mode in (("post_norm", "const") if name == "physics" else ("const",)):
                        def records(VV, HR, ns):
                            rec = {}
                            for n in ns:
                                p_full, p_L = point(n, "full", VV), point(n, Ls, VV)
                                y = np.log1p(p_full[1] if ref_kind == "self" else HR[n])
                                if mode == "post_norm":
                                    h = VV[n][Ls]["hypo"][:, jd[fold.get(n, -1)]]
                                    lo, hi = np.log1p(min(h[iq[0]], h[iq[1]])), np.log1p(max(h[iq[0]], h[iq[1]]))
                                    w = max(hi - lo, 0.05)
                                else:
                                    lo = hi = np.log1p(p_L[1])
                                    w = 1.0
                                rec[n] = (lo, hi, w, y, max(lo - y, y - hi) / w)
                            return rec

                        rec = records(V, hypo_ref, pats)
                        cal_sc = [r[4] for r in records(calV, cal_hypo, cal_pats).values()] if ext else None
                        cov, wid, cert = [], [], []
                        for n in pats:
                            sc = cal_sc if ext else [rec[m][4] for m in pats if fold[m] != fold[n]]
                            q = float(np.quantile(sc, min(1.0, a.level * (1 + 1 / len(sc)))))
                            lo, hi, w, y, _ = rec[n]
                            lo, hi = lo - q * w, hi + q * w
                            cov.append(lo <= y <= hi)
                            wid.append(float(np.expm1(hi) - np.expm1(max(lo, 0.0))))
                            cert.append(d3(pt[n][0], float(np.expm1(max(lo, 0.0)))) == d3(pt[n][0], float(np.expm1(hi))))
                        cert = np.array(cert)
                        out.setdefault("sel", {}).setdefault(name, {})[f"{ref_kind}_{mode}_{Ls}"] = {
                            str(n): [int(cert[i]), float(wid[i])] for i, n in enumerate(pats)}
                        res[f"{ref_kind}_{mode}"] = {
                            "cover": float(np.mean(cov)), "width_median_ml": float(np.median(wid)), "certain": float(cert.mean()),
                            "flip_certain": float(flip[cert].mean()) if cert.any() else None,
                            "flip_uncertain": float(flip[~cert].mean()) if (~cert).any() else None,
                            "agree_certain": float(agree[cert].mean()) if cert.any() else None}
            out[name][Ls] = res
            print(name, Ls, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}),
                  *[f"\n   {k}: " + json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                    for k, v in res.items() if isinstance(v, dict)], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("train", "infer", "eval"))
    p.add_argument("args", nargs="+")
    p.add_argument("--masks", action="store_true")
    p.add_argument("--phys-weight", type=float, default=1.0)
    p.add_argument("--adapt-cnn", help="train: cnn_export dir of the new centre (label-free adaptation)")
    p.add_argument("--adapt-x", help="train: twin-target dir of the new centre")
    p.add_argument("--adapt-folds", help="train: json patient -> half (0/1) of the new centre")
    p.add_argument("--adapt-fold", type=int, default=0, help="train: the test half, never used for adaptation")
    p.add_argument("--pseudo", help="train: work dir of the source-trained control used as the teacher of pseudo-masks")
    p.add_argument("--reliable", action="store_true", help="train: weight the twin loss by trel (ROIs with beta <= 1 s) instead of tmask")
    p.add_argument("--swap", choices=("aif", "ctx", "both"), help="infer diagnostic: template AIF / tissue curve")
    p.add_argument("--swap-from", default="at2d24x")
    p.add_argument("--shift", type=int, default=0, help="infer: drop the first k frames of every input (a later start)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--fold", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", type=float, default=0.8)
    p.add_argument("--external", help="sub-directory of work_dir with the predictions of another cohort")
    p.add_argument("--ext-refs")
    a = p.parse_args()
    if a.stage == "eval":
        a.work, a.refs, a.out = Path(a.args[0]), a.args[1], a.args[2]
        evaluate(a)
        return
    a.cnn, a.x, a.folds, a.work = a.args[0], a.args[1], a.args[2], Path(a.args[3])
    (train if a.stage == "train" else infer)(a)


if __name__ == "__main__":
    main()
