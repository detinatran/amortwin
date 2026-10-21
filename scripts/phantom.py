"""Phantom with known parameters: AmorTwin and the twin against the ground truth under truncation.

Stage 1, CPU pool, one thread per worker:
python scripts/phantom.py simulate <ISLES2024_Training_s3> <twin_out_s3> <work_dir> --folds scripts/learned_folds24.json
       [--patients 40] [--per-patient 600] [--lengths full 25 30 35] [--workers 16]
  Per patient, voxels sampled as export_pairs.py (a third inside the final infarct when possible), with
  its curve preprocessing, context curve, AIF and real targets. Each voxel's model is the twin's voxel
  model (hemopic_plus.npde.voxel, n_cells = 1, the ROI parameters of the full-scan twin fit held fixed);
  its fit on the real curve gives x_fit = (log V, log Cin, delay). The true parameters are
  x_true = x_fit + jitter (ln V and so ln MTT U[-0.3, 0.3], ln Cin and so ln CBV U[-0.2, 0.2], delay
  U[-1.5, 1.5] s), the phantom curve is the model curve of x_true plus white Gaussian noise of the SD of
  the voxel's real residual, preprocessed as the real curves (smooth_curves). Ground truth in the units
  of the AmorTwin targets, by the shift from the real target: ln_rel_MTT + dx0, delay + dx2,
  ln_rel_CBV + dx1 (CBV is proportional to Cin at fixed ROI) and Tmax = delay + residue peak (ROI
  dispersion, true MTT). The twin estimate per length is the voxel fit on the first L frames of the
  phantom curve, with the same ROI (oracle ROI: it isolates the voxel-level information loss and
  favours the twin), mapped to the target units the same way. Writes <work_dir>/sim/training_X.npz.

Stage 3, CPU:
python scripts/phantom.py conformal <work_dir> [--lengths full 25 30 35]
  Conformal calibration against the ground truth instead of the twin: per length and target, the
  scale of AmorTwin's Gaussian SD is the 80 % quantile of z = |truth - mean| / SD over the phantom
  patients of the other folds (learned_folds24.json) divided by the Gaussian quantile; reports the
  coverage of the truth on fold k by class, the median width, and the ratio of the phantom scale to
  the twin-calibrated one (calib.json). Mondrian variants (one scale per group, groups known at
  inference): "sd", quintiles of the predicted SD of the target (cut points from the calibration
  folds); "tmax", the predicted Tmax (< 2, 2-4, 4-6, > 6 s). Writes <work_dir>/conformal.json.

Stage 2, GPU, one process:
python scripts/phantom.py score <work_dir> <prob_weights_dir> <calib.json> [--lengths full 25 30 35]
  AmorTwin (the patient's held-out fold model) on the phantom curves at every length: mean and the
  80 % interval of its Gaussian posterior, raw and with the cross-conformal scale of the fold
  (calib.json, fitted on the real curves against the twin). Reports per length, target and class
  (all; true Tmax > 6 s or not; vendor Tmax > 6 s or < 4 s; ROI dispersion beta > 1 s; true Tmax > 6 s with
  beta <= 1 s): MAE, bias and r against the truth for AmorTwin and the twin, and the
  coverage of the truth by AmorTwin's calibrated interval. Writes <work_dir>/phantom.json.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "HemoPIC_plus"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TARGETS = ("ln_rel_MTT", "delay_s", "ln_rel_CBV", "tmax_s")
TMAX = 70
Z80 = 1.2816
LAB = "roi_labels_seg4_within_kmeans.nii.gz"
JITTER = (0.3, 0.2, 1.5)          # half-widths for (log V, log Cin, delay)


def residue_peak(beta, mtt):
    beta, mtt = np.maximum(beta, 1e-3), np.maximum(mtt, 1e-3)
    close = np.abs(mtt - beta) < 1e-3 * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta * mtt * np.log(mtt / beta) / (mtt - beta)
    return np.where(close, beta, t)


def to_targets(tgt_real, mtt_real, beta, x, x_fit):
    """Targets of parameters x, by their shift from the voxel's fit on its real curve."""
    d = x - x_fit
    mtt = mtt_real * np.exp(d[:, 0])
    delay = tgt_real[:, 1] + d[:, 2]
    return np.stack([tgt_real[:, 0] + d[:, 0], delay, tgt_real[:, 2] + d[:, 1], delay + residue_peak(beta, mtt)], 1)


def simulate(args):
    from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, squeeze_ctc
    from hemopic_plus.npde.voxel import RoiModel, fit_voxels, smooth_curves
    from hemopic_plus.uq.data import load_subject as load_fit
    from hemopic_plus.uq.torch_model import DTYPE

    data_root, duo, work, n, per, lengths, seed = args
    od = work / "sim" / f"training_{n}.npz"
    if od.exists():
        return n, "done"
    torch.set_num_threads(1)
    d = data_root / f"training_{n}"
    fit_dir = duo / "isles24_full_aif" / f"training_{n}"
    vd = duo / "isles24_full_aif__vox_tank" / f"training_{n}"
    cbv_f = duo / "isles24_full_aif_idvox_osvd_nc" / f"training_{n}" / "HemoPIC_CBV.nii.gz"
    if not ((vd / "VOX_MTT.nii.gz").exists() and cbv_f.exists() and (fit_dir / "fit_curves.npz").exists()):
        return n, "missing"
    fit = load_fit(fit_dir)
    cfg = fit.cfg
    lab = np.rint(nib.load(str(fit_dir / LAB)).get_fdata()).astype(int)
    rois = {int(r): i for i, r in enumerate(fit.roi_ids)}
    disp = np.full(lab.max() + 1, np.nan)
    for rid, i in rois.items():
        disp[rid] = float(np.exp(fit.theta_fit[i].numpy()[6]))
    ctc = squeeze_ctc(load_nifti(find_nii(d, "CTC_from_MR_4DPWI"))[1])
    seg = load_nifti(d / SEG4_REL)[1].astype(int)
    ot = load_nifti(find_nii(d, "OT"))[1] > 0
    mtt = nib.load(str(vd / "VOX_MTT.nii.gz")).get_fdata()
    delay = nib.load(str(vd / "VOX_delay.nii.gz")).get_fdata()
    cbv = nib.load(str(cbv_f)).get_fdata()
    tissue = np.isin(seg, (1, 2))
    beta = np.where(lab > 0, disp[np.clip(lab, 0, None)], np.nan)
    ok = (tissue & np.isfinite(mtt) & (mtt > 0) & np.isfinite(delay) & np.isfinite(cbv) & (cbv > 0) & np.isfinite(beta)
          & np.all(np.isfinite(ctc), axis=3))
    if ok.sum() < 100:
        return n, "few voxels"
    T = int(fit.t.numel())
    ctc = ctc[..., :T]
    med_mtt, med_cbv = float(np.median(mtt[ok])), float(np.median(cbv[ok]))
    allc = smooth_curves(ctc[ok], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
    ctx = np.zeros(TMAX, np.float32)
    ctx[:min(allc.shape[1], TMAX)] = np.median(allc, axis=0)[:TMAX]
    a = np.load(fit_dir / "fit_curves.npz")["cart_norm"]
    aif = np.zeros(TMAX, np.float32)
    aif[:min(a.size, TMAX)] = a[:TMAX]
    rng = np.random.default_rng(seed + n)
    ok &= np.isin(lab, list(rois))
    les, rest = np.argwhere(ok & ot), np.argwhere(ok & ~ot)
    k_les = min(len(les), per // 3)
    pick = np.concatenate([les[rng.choice(len(les), k_les, replace=False)] if k_les else np.zeros((0, 3), int),
                           rest[rng.choice(len(rest), min(len(rest), per - k_les), replace=False)]])
    idx = tuple(pick.T)
    mtt_r, delay_r, beta_r = mtt[idx], delay[idx], beta[idx]
    sys.path.insert(0, str(ROOT / "external" / "duofusion"))
    from duofusion.data import load_subject as load_case
    vmap = load_case(data_root, n).maps.get("Tmax")
    vendor = vmap[idx].astype(np.float32) if vmap is not None else np.full(len(pick), np.nan, np.float32)
    tgt_real = np.stack([np.log(mtt_r / med_mtt), delay_r, np.log(cbv[idx] / med_cbv),
                         delay_r + residue_peak(beta_r, mtt_r)], 1)
    N = len(pick)
    rids = lab[idx]
    curve = np.zeros((N, TMAX), np.float32)
    curve_real = np.zeros((N, TMAX), np.float32)
    truth = np.zeros((N, 4))
    twin = {L: np.full((N, 4), np.nan) for L in lengths}
    w = fit.w[0].numpy()
    lo_d, hi_d = (float(b) for b in cfg.delay_bounds)
    for rid in np.unique(rids):
        sel = np.nonzero(rids == rid)[0]
        th = np.r_[fit.theta_fit[rois[int(rid)]].numpy(), 0.0]
        model = RoiModel(cfg, fit.t.numpy(), fit.dt, fit.shape.numpy(), th, n_cells=1)
        raw = ctc[tuple(pick[sel].T)].astype(np.float64)
        c = smooth_curves(raw, cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
        curve_real[sel, :min(T, TMAX)] = c[:, :TMAX]
        x_fit, _, _ = fit_voxels(model, c, w)
        x_fit = np.asarray(x_fit, np.float64)
        with torch.no_grad():
            cm_fit = model.simulate(torch.as_tensor(x_fit, dtype=DTYPE))[1].numpy()
        base = raw[:, :max(int(cfg.baseline_frames), 1)].mean(1, keepdims=True)
        sigma = np.std((raw - base) - cm_fit, axis=1)
        x_true = x_fit + rng.uniform(-1, 1, (len(sel), 3)) * np.array(JITTER)
        x_true[:, 2] = np.clip(x_true[:, 2], lo_d, hi_d)
        with torch.no_grad():
            cm = model.simulate(torch.as_tensor(x_true, dtype=DTYPE))[1].numpy()
        syn = cm + rng.standard_normal(cm.shape) * sigma[:, None]
        cs = smooth_curves(syn, cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
        curve[sel, :min(T, TMAX)] = cs[:, :TMAX]
        args_t = (tgt_real[sel], mtt_r[sel], beta_r[sel])
        truth[sel] = to_targets(*args_t, x_true, x_fit)
        for L in lengths:
            k = T if L == "full" else min(T, int(np.floor(float(L) / fit.dt)) + 1)
            mk = RoiModel(cfg, fit.t.numpy()[:k], fit.dt, fit.shape.numpy()[:k], th, n_cells=1)
            xh, _, _ = fit_voxels(mk, cs[:, :k], w[:k])
            twin[L][sel] = to_targets(*args_t, np.asarray(xh, np.float64), x_fit)
    od.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od, pid=n, curve=curve, ctx=ctx, aif=aif, length=min(T, TMAX), tissue=seg[idx].astype(np.int8),
                        lesion=ot[idx], truth=truth, real=tgt_real, curve_real=curve_real, vendor_tmax=vendor, beta=beta_r, med_mtt=med_mtt, lengths=np.array(lengths),
                        **{f"twin_{L}": twin[L] for L in lengths})
    return n, f"ok ({N} voxels)"


def stats(est, y):
    e = est - y
    return {"mae": float(np.abs(e).mean()), "bias": float(e.mean()), "r": float(np.corrcoef(est, y)[0, 1])}


def score(a):
    from amortized_model import AmorTwin, make_inputs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    cal = json.loads(a.calib.read_text())
    K = len(TARGETS)
    models = {}
    for f in sorted(a.weights.glob("amortized_fold*.pt")):
        ck = torch.load(f, map_location=dev)
        m = AmorTwin(n_out=2 * K).to(dev)
        m.load_state_dict(ck["state"])
        models[int(f.stem[-1])] = (m.eval(), ck["mu"].to(dev).float(), ck["sd"].to(dev).float())
    sims = [np.load(f) for f in sorted((a.work / "sim").glob("training_*.npz"))]
    Y = np.concatenate([s["truth"] for s in sims])
    hypo = Y[:, 3] > 6
    vend = np.concatenate([s["vendor_tmax"] for s in sims])
    bbig = np.concatenate([s["beta"] for s in sims]) > 1.0
    classes = (("all", np.ones(len(Y), bool)), ("hypo", hypo), ("normal", ~hypo), ("vendor_tmax_gt6", vend > 6),
               ("vendor_tmax_lt4", vend < 4), ("beta_gt1", bbig), ("hypo_beta_le1", hypo & ~bbig))
    out = {"patients": len(sims), "voxels": int(len(Y)), "hypo_voxels": int(hypo.sum()), "jitter": JITTER, "lengths": {}}
    for L in a.lengths:
        mean, sd, scale = [], [], []
        for s in sims:
            k = folds[int(s["pid"])]
            m, mu, sdv = models[k]
            N = len(s["truth"])
            ln = int(s["length"]) if L == "full" else min(int(s["length"]), int(L))
            x = make_inputs(torch.as_tensor(s["curve"]), torch.as_tensor(np.repeat(s["ctx"][None], N, 0)),
                            torch.as_tensor(np.repeat(s["aif"][None], N, 0)), torch.full((N,), ln),
                            torch.as_tensor(s["tissue"].astype(np.int64))).to(dev)
            with torch.no_grad():
                o = m(x).float()
            mean.append((o[:, :K] * sdv + mu).cpu().numpy())
            sd.append((torch.exp(0.5 * o[:, K:].clamp(-9.0, 6.0)) * sdv).cpu().numpy())
            scale.append(np.repeat([[cal["scales"][L][t]["0.8"][str(k)] for t in TARGETS]], N, 0))
        mean, sd, scale = np.concatenate(mean), np.concatenate(sd), np.concatenate(scale)
        tw = np.concatenate([s[f"twin_{L}"] for s in sims])
        np.savez_compressed(a.work / f"pred_{L}.npz", mean=mean, sd=sd, scale=scale, truth=Y, twin=tw, vendor_tmax=vend,
                            beta=np.concatenate([s["beta"] for s in sims]),
                            pid=np.concatenate([np.full(len(s["truth"]), int(s["pid"])) for s in sims]))
        z = np.abs(Y - mean) / sd
        res = {}
        for j, t in enumerate(TARGETS):
            r = {}
            for cls, msk in classes:
                ok = msk & np.isfinite(tw[:, j])
                r[cls] = {"amortwin": stats(mean[msk, j], Y[msk, j]), "twin": stats(tw[ok, j], Y[ok, j]),
                          "cover80_raw": float(np.mean(z[msk, j] <= Z80)),
                          "cover80_cal": float(np.mean(z[msk, j] <= Z80 * scale[msk, j])),
                          "width80_cal_median": float(np.median(2 * Z80 * scale[msk, j] * sd[msk, j]))}
            res[t] = r
        out["lengths"][L] = res
        out["class_voxels"] = {c: int(m.sum()) for c, m in classes}
        for cls, _ in classes:
            print(L.rjust(4), cls.ljust(16), " | ".join(
                f"{t[:7]} amor {res[t][cls]['amortwin']['mae']:.2f} twin {res[t][cls]['twin']['mae']:.2f} cov {res[t][cls]['cover80_cal']:.2f}"
                for t in TARGETS), flush=True)
    (a.work / "phantom.json").write_text(json.dumps(out, indent=1))


def conformal(a):
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    out = {}
    for L in a.lengths:
        d = np.load(a.work / f"pred_{L}.npz")
        Y, mean, sd, s24 = d["truth"], d["mean"], d["sd"], d["scale"]
        fold = np.array([folds[int(p)] for p in d["pid"]])
        z = np.abs(Y - mean) / sd
        hypo = Y[:, 3] > 6
        classes = {"all": np.ones(len(Y), bool), "vendor_tmax_gt6": d["vendor_tmax"] > 6, "vendor_tmax_lt4": d["vendor_tmax"] < 4,
                   "beta_gt1": d["beta"] > 1, "hypo_beta_le1": hypo & (d["beta"] <= 1)}
        def scales(group):
            sc = np.zeros_like(z)
            for k in range(5):
                te = fold == k
                for j in range(z.shape[1]):
                    g = group(j, ~te)
                    for v in np.unique(g):
                        cal, tst = ~te & (g == v), te & (g == v)
                        if tst.any():
                            sc[tst, j] = np.quantile(z[cal, j] if cal.sum() >= 50 else z[~te, j], 0.8) / Z80
            return sc

        def sd_bins(j, cal):
            cuts = np.quantile(sd[cal, j], [0.2, 0.4, 0.6, 0.8])
            return np.digitize(sd[:, j], cuts)

        variants = {"global": scales(lambda j, cal: np.zeros(len(Y), int)), "sd": scales(sd_bins),
                    "tmax": scales(lambda j, cal: np.digitize(mean[:, 3], [2.0, 4.0, 6.0]))}
        res = {}
        for j, t in enumerate(TARGETS):
            res[t] = {"scale_phantom_over_twin": float(np.median(variants["global"][:, j] / s24[:, j]))}
            for c, m in classes.items():
                r = {"cover80_twin_cal": float(np.mean(z[m, j] <= Z80 * s24[m, j])),
                     "width80_twin_cal_median": float(np.median(2 * Z80 * s24[m, j] * sd[m, j]))}
                for vn, sc in variants.items():
                    r[f"cover80_{vn}"] = float(np.mean(z[m, j] <= Z80 * sc[m, j]))
                    r[f"width80_{vn}_median"] = float(np.median(2 * Z80 * sc[m, j] * sd[m, j]))
                res[t][c] = r
        out[L] = res
        for t in TARGETS:
            r = res[t]
            for vn in variants:
                print(L.rjust(4), t.ljust(10), vn.ljust(6), " | ".join(
                    f"{c} {r[c][f'cover80_{vn}']:.2f} w {r[c][f'width80_{vn}_median']:.2f}" for c in classes), flush=True)
    (a.work / "conformal.json").write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("simulate", "score", "conformal"))
    p.add_argument("args", nargs="+", type=Path)
    p.add_argument("--folds", type=Path, default=Path(__file__).resolve().parent / "learned_folds24.json")
    p.add_argument("--patients", type=int, default=40)
    p.add_argument("--per-patient", type=int, default=600)
    p.add_argument("--lengths", nargs="+", default=["full", "25", "30", "35"])
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.stage == "simulate":
        data_root, duo, work = a.args
        nums = sorted(int(k) for k in json.loads(a.folds.read_text()))
        nums = [n for n in nums if (duo / "isles24_full_aif__vox_tank" / f"training_{n}" / "VOX_MTT.nii.gz").exists()]
        nums = [int(v) for v in np.random.default_rng(0).permutation(nums)][:a.patients]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for n, msg in ex.map(simulate, [(data_root, duo, work, n, a.per_patient, a.lengths, 0) for n in nums]):
                print(n, msg, flush=True)
    elif a.stage == "conformal":
        (a.work,) = a.args
        conformal(a)
    else:
        a.work, a.weights, a.calib = a.args
        score(a)


if __name__ == "__main__":
    main()
