"""Probabilistic AmorTwin: conformal calibration of the amortized posterior and its propagation to
volumes and to the DEFUSE 3 decision of truncated scans (towards a calibrated, decision-level twin).

The network is AmorTwin with a Gaussian head (amortized_train.py --prob --trunc-min 25): a mean and
a log-variance for ln rel. MTT, delay, ln rel. CBV and the twin Tmax, at any scan length.

python scripts/prob_amortwin.py calib   <pairs.npz> <weights_dir> <work_dir>
  GPU. Held-out z = |y - mean| / sd of every fold model on its own fold, for L = full, 25, 30, 35 s.
  Cross-conformal: for fold k, the scale s[L][target][level] is the level-quantile of z over the
  patients of the other folds divided by the Gaussian quantile; reports the coverage of the raw and
  of the calibrated intervals on fold k. Writes <work_dir>/calib.json.
python scripts/prob_amortwin.py maps    <data_root> <twin_out> <weights_dir> <work_dir>
  GPU. Per ISLES'24 patient (its held-out fold model) and length: maps of the mean and SD of Tmax and
  of the mean relative CBF (= CBV / MTT), inputs as amortized_volumes.py (AIF from select_aif on the
  cut scan). Writes <work_dir>/maps/training_X.npz.
python scripts/prob_amortwin.py volumes <data_root> <twin_out> <work_dir> [--workers 10]
  CPU pool. Tmax maps at the quantiles q of Q_GRID of the calibrated posterior, comonotone across
  voxels (every voxel at the same quantile: the conservative, fully correlated case; a common shift
  cancels in the relative Tmax), each smoothed and thresholded exactly as amortized_volumes.py;
  core from the mean CBF. Writes <work_dir>/vol/<L>/training_X.npz (core_grid, hypo_grid per q).
python scripts/prob_amortwin.py score2  <results_dir> <work_dir>
  CPU. Volume-level conformal (CQR on log1p volumes), per scan length and cross-fitted by patient
  folds: the score of a patient is max(log1p lo - log1p ref, log1p ref - log1p hi) for the raw
  [q 0.1, q 0.9] hypoperfusion interval; the 80 % quantile of the scores of the other folds widens
  (or narrows) the interval of fold k. Reports the coverage and width of the calibrated interval;
  a patient is "certain" when the DEFUSE 3 decision is the same at both calibrated ends (core at the
  posterior median), and the risk-coverage curve ranks patients by |P(eligible) - 0.5| and reports the
  agreement with the vendor decision and the flip rate among the most confident 50/70/90/100 %.
  Writes <work_dir>/score2.json.
python scripts/prob_amortwin.py score   <results_dir> <work_dir>
  Thresholds of the AmorTwin method cross-fitted on the full scans (truncation_accuracy.py). Per
  length: coverage of the 80 % hypoperfusion interval [q 0.1, q 0.9] of the vendor reference volume,
  P(DEFUSE 3 eligible) = share of the quantile grid that is eligible, its Brier score against the
  vendor decision of the full scan, and whether uncertain patients (0.1 < P < 0.9) are those whose
  point decision flips. Writes <work_dir>/score.json.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = Path(__file__).resolve().parent
DF = HERE.parent.parent / "duofusion"
for p_ in (DF, DF / "external" / "HemoPIC_plus", DF / "scripts", HERE):
    sys.path.insert(0, str(p_))

import numpy as np  # noqa: E402

LENGTHS = ("full", "25", "30", "35")
LEVELS = (0.8, 0.9)
ZQ = {0.8: 1.2816, 0.9: 1.6449}
Q_GRID = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
TARGETS = ("ln_rel_MTT", "delay_s", "ln_rel_CBV", "tmax_s")


def load_models(weights, dev):
    import torch
    from amortized_model import AmorTwin
    out = {}
    for f in sorted(Path(weights).glob("amortized_fold*.pt")):
        k = int(f.stem.replace("amortized_fold", ""))
        ck = torch.load(f, map_location=dev)
        m = AmorTwin(n_out=2 * len(TARGETS)).to(dev)
        m.load_state_dict(ck["state"])
        out[k] = (m.eval(), ck["mu"].to(dev).float(), ck["sd"].to(dev).float())
    return out


# ------------------------------------------------------------------ calib (GPU)
def calib(a):
    import torch
    from amortized_model import make_inputs
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
    PID = z["pid"].astype(np.int64)
    models = load_models(a.weights, dev)
    K = len(TARGETS)
    zs = {}          # zs[L] = (n, K) array of |y - mean| / sd for every voxel, by its held-out fold model
    for L in LENGTHS:
        zz = np.empty((len(FOLD), K), np.float32)
        for k, (m, mu, sd) in models.items():
            te = np.flatnonzero(FOLD == k)
            for s in range(0, len(te), 65536):
                i = torch.as_tensor(te[s:s + 65536], device=dev)
                ln = LEN[i] if L == "full" else torch.clamp(LEN[i], max=int(L))
                with torch.no_grad():
                    out = m(make_inputs(CUR[i].float(), CTX[P[i]], AIF[P[i]], ln, TIS[i])).float()
                mean = out[:, :K] * sd + mu
                psd = torch.exp(0.5 * out[:, K:].clamp(-9.0, 6.0)) * sd
                zz[te[s:s + 65536]] = ((Y[i] - mean).abs() / psd).cpu().numpy()
        zs[L] = zz
        print("calib", L, flush=True)
    res = {"levels": LEVELS, "scales": {}, "coverage": {}}
    for L in LENGTHS:
        res["scales"][L], res["coverage"][L] = {}, {}
        for j, t in enumerate(TARGETS):
            sc, cov = {}, {"raw": {}, "conformal": {}}
            for lev in LEVELS:
                sk, raw, cal = {}, [], []
                for k in models:
                    tr, te = FOLD != k, FOLD == k
                    # one quantile per calibration patient pool (voxels of the other folds' patients)
                    s_k = float(np.quantile(zs[L][tr, j], lev)) / ZQ[lev]
                    sk[str(k)] = s_k
                    raw.append(float(np.mean(zs[L][te, j] <= ZQ[lev])))
                    cal.append(float(np.mean(zs[L][te, j] <= ZQ[lev] * s_k)))
                sc[str(lev)] = sk
                cov["raw"][str(lev)] = float(np.mean(raw))
                cov["conformal"][str(lev)] = float(np.mean(cal))
                # patient-level spread of the calibrated coverage (does it hold per patient?)
                per_pat = [float(np.mean(zs[L][(PID == p), j] <= ZQ[lev] * sk[str(int(FOLD[PID == p][0]))]))
                           for p in np.unique(PID)]
                cov["conformal"][f"{lev}_patient_p10_p90"] = [float(np.percentile(per_pat, 10)), float(np.percentile(per_pat, 90))]
            res["scales"][L][t] = sc
            res["coverage"][L][t] = cov
    Path(a.work).mkdir(parents=True, exist_ok=True)
    (Path(a.work) / "calib.json").write_text(json.dumps(res, indent=1))
    for L in LENGTHS:
        print(L.rjust(4), " | ".join(
            f"{t[:7]} raw {res['coverage'][L][t]['raw']['0.8']:.2f}/{res['coverage'][L][t]['raw']['0.9']:.2f}"
            f" -> conf {res['coverage'][L][t]['conformal']['0.8']:.2f}/{res['coverage'][L][t]['conformal']['0.9']:.2f}"
            f" (x{np.mean(list(res['scales'][L][t]['0.8'].values())):.1f})" for t in TARGETS), flush=True)


# ------------------------------------------------------------------ maps (GPU)
def maps(a):
    import torch
    from amortized_model import TMAX, make_inputs
    from duofusion.data import load_subject
    from hemopic_plus.config import load_config
    from hemopic_plus.io import SEG4_REL, load_nifti
    from hemopic_plus.npde.voxel import smooth_curves
    from time_amortized import auto_aif
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(DF / "external" / "HemoPIC_plus" / "configs" / "isles24_full_aif.yaml", {"data_root": str(a.data_root)})
    folds = {int(k): v for k, v in json.loads((HERE / "learned_folds24.json").read_text()).items()}
    models = load_models(a.weights, dev)
    K = len(TARGETS)
    od = Path(a.work) / "maps"
    od.mkdir(parents=True, exist_ok=True)
    nums = sorted(int(d.name.split("_")[1]) for d in (Path(a.twin_out) / "isles24_full_aif").glob("training_*"))
    for n in nums:
        f = od / f"training_{n}.npz"
        if f.exists() or n not in folds:
            continue
        fit_dir = Path(a.twin_out) / "isles24_full_aif" / f"training_{n}"
        if not (fit_dir / "run_info_core.json").exists():
            continue
        dt = float(json.loads((fit_dir / "run_info_core.json").read_text())["dt_seconds"])
        s = load_subject(a.data_root, n)
        ctc_full = s.ctc()
        seg4 = load_nifti(Path(a.data_root) / f"training_{n}" / SEG4_REL)[1].astype(np.int32)
        m, mu, sd = models[folds[n]]
        out = {}
        for L in LENGTHS:
            keep = ctc_full.shape[3] if L == "full" else min(ctc_full.shape[3], int(np.floor(float(L) / dt)) + 1)
            ctc = ctc_full[..., :keep]
            gmwm = np.isin(seg4, (1, 2)) & np.all(np.isfinite(ctc), axis=3)
            aif_full, _ = auto_aif(ctc, seg4, cfg, n, dt)
            c = smooth_curves(ctc[gmwm], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
            T = min(c.shape[1], TMAX)
            curve = np.zeros((len(c), TMAX), np.float32)
            curve[:, :T] = c[:, :T]
            ctx = np.zeros(TMAX, np.float32)
            ctx[:T] = np.median(c, axis=0)[:T]
            aif = np.zeros(TMAX, np.float32)
            aif[:min(aif_full.size, TMAX)] = aif_full[:TMAX]
            N = len(curve)
            res = []
            with torch.no_grad():
                for s0 in range(0, N, 65536):
                    sl = slice(s0, s0 + 65536)
                    x = make_inputs(curve[sl], np.broadcast_to(ctx, (len(curve[sl]), TMAX)),
                                    np.broadcast_to(aif, (len(curve[sl]), TMAX)), np.full(len(curve[sl]), T),
                                    seg4[gmwm][sl]).to(dev)
                    o = m(x).float()
                    res.append(torch.cat([o[:, :K] * sd + mu, torch.exp(0.5 * o[:, K:].clamp(-9.0, 6.0)) * sd], 1).cpu().numpy())
            r = np.concatenate(res)
            for name, col in (("tmax_mean", r[:, 3]), ("tmax_sd", r[:, K + 3]), ("cbf_rel", np.exp(r[:, 2] - r[:, 0]))):
                v = np.full(seg4.shape, np.nan, np.float32)
                v[gmwm] = col
                out[f"{name}_{L}"] = v
        np.savez_compressed(f, **out)
        print("maps", n, flush=True)


# ------------------------------------------------------------------ volumes (CPU)
def vol_one(args):
    data_root, twin_out, work, n, L, scales = args
    import nibabel as nib
    from duofusion.data import load_subject
    from amortized_volumes import grids_from_maps
    from scipy.stats import norm
    of = Path(work) / "vol" / L / f"training_{n}.npz"
    if of.exists():
        return n, L, "done"
    mf = Path(work) / "maps" / f"training_{n}.npz"
    fit_dir = Path(twin_out) / "isles24_full_aif" / f"training_{n}"
    if not mf.exists():
        return n, L, "no maps"
    z = np.load(mf)
    s = load_subject(data_root, n)
    lab = np.rint(nib.load(str(fit_dir / "roi_labels_seg4_within_kmeans.nii.gz")).get_fdata()).astype(int)
    mean, sd, cbf = z[f"tmax_mean_{L}"], z[f"tmax_sd_{L}"], z[f"cbf_rel_{L}"]
    brain = (s.tissue > 0) & (lab > 0) & np.isfinite(mean)
    core, hypo = [], []
    for q in Q_GRID:
        zq = norm.ppf(q)
        # calibrated SD: the conformal scale of the nearer level (0.8 for |z| < 1.47, else 0.9)
        sc = scales["0.8"] if abs(zq) < 1.47 else scales["0.9"]
        g = grids_from_maps(s, brain, cbf, mean + zq * sc * sd, 3.0)
        core.append(g["core_grid"])
        hypo.append(g["hypo_grid"])
    of.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(of, q=Q_GRID, core_grid=np.array(core), hypo_grid=np.array(hypo),
                        tau_grid=g["tau_grid"], dt_grid=g["dt_grid"])
    return n, L, "ok"


def volumes(a):
    cal = json.loads((Path(a.work) / "calib.json").read_text())
    folds = {int(k): v for k, v in json.loads((HERE / "learned_folds24.json").read_text()).items()}
    nums = sorted(int(f.stem.split("_")[1]) for f in (Path(a.work) / "maps").glob("training_*.npz"))
    jobs = []
    for L in LENGTHS:
        for n in nums:
            k = str(folds[n])
            jobs.append((a.data_root, a.twin_out, a.work, n, L,
                         {lev: cal["scales"][L]["tmax_s"][lev][k] for lev in ("0.8", "0.9")}))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(vol_one, jobs)):
            if i % 50 == 0:
                print("volumes", i, len(jobs), r, flush=True)


# ------------------------------------------------------------------ score (CPU)
def score(a):
    import truncation_accuracy as ta
    from weighted_fusion import references
    R = Path(a.results)
    core_ref, hypo_ref, dec_ref, groups = references(R, "24")
    full = ta.grids(R, "amortized", "24", "full")       # AmorTwin (point model) full-scan grids for the threshold choice
    W = Path(a.work) / "vol"
    pats = sorted(n for n in full if n in core_ref and (W / "full" / f"training_{n}.npz").exists())
    fold = ta.folds({n: groups[n] for n in pats})
    # thresholds cross-fitted on the median (q = 0.5) full-scan volumes of the probabilistic model
    med = {n: np.load(W / "full" / f"training_{n}.npz") for n in pats}
    qi = int(np.argmin(np.abs(Q_GRID - 0.5)))
    lc = {n: np.log1p(med[n]["core_grid"][qi]) for n in pats}
    lh = {n: np.log1p(med[n]["hypo_grid"][qi]) for n in pats}
    choice = {}
    for k in range(5):
        tr = [n for n in pats if fold[n] != k]
        ec = np.median(np.abs(np.array([lc[n] for n in tr]) - np.log1p([core_ref[n] for n in tr])[:, None]), axis=0)
        eh = np.median(np.abs(np.array([lh[n] for n in tr]) - np.log1p([hypo_ref[n] for n in tr])[:, None]), axis=0)
        choice[k] = (int(np.argmin(ec)), int(np.argmin(eh)))
    i10, i90 = int(np.argmin(np.abs(Q_GRID - 0.1))), int(np.argmin(np.abs(Q_GRID - 0.9)))
    out = {}
    for L in LENGTHS:
        rows = []
        for n in pats:
            f = W / L / f"training_{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            i, j = choice[fold[n]]
            core = float(z["core_grid"][qi][i])
            hyp = z["hypo_grid"][:, j]                    # per quantile
            dec_q = [ta.defuse3(core, max(core, float(h))) for h in hyp]
            p_elig = float(np.mean(dec_q))
            d_point = ta.defuse3(core, max(core, float(hyp[qi])))
            zf = np.load(W / "full" / f"training_{n}.npz")
            cf, hf = float(zf["core_grid"][qi][i]), float(zf["hypo_grid"][qi][j])
            d_full = ta.defuse3(cf, max(cf, hf))
            lo, hi = sorted((float(hyp[i10]), float(hyp[i90])))
            rows.append({"cover80": lo <= hypo_ref[n] <= hi, "width80": hi - lo, "p": p_elig,
                         "ref": dec_ref[n], "flip": d_point != d_full, "agree": d_point == dec_ref[n],
                         "uncertain": 0.1 < p_elig < 0.9})
        p = np.array([r["p"] for r in rows])
        ref = np.array([r["ref"] for r in rows], float)
        flip = np.array([r["flip"] for r in rows])
        unc = np.array([r["uncertain"] for r in rows])
        agree = np.array([r["agree"] for r in rows])
        res = {"n": len(rows), "hypo_cover80": float(np.mean([r["cover80"] for r in rows])),
               "hypo_width80_median_ml": float(np.median([r["width80"] for r in rows])),
               "brier_vs_vendor": float(np.mean((p - ref) ** 2)),
               "flip_rate": float(flip.mean()), "agree": float(agree.mean()),
               "uncertain_share": float(unc.mean()),
               "flip_rate_uncertain": float(flip[unc].mean()) if unc.any() else None,
               "flip_rate_certain": float(flip[~unc].mean()) if (~unc).any() else None,
               "agree_certain": float(agree[~unc].mean()) if (~unc).any() else None}
        out[L] = res
        print(L.rjust(4), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    (Path(a.work) / "score.json").write_text(json.dumps(out, indent=1))


def score2(a):
    import truncation_accuracy as ta
    from weighted_fusion import references
    R = Path(a.results)
    core_ref, hypo_ref, dec_ref, groups = references(R, "24")
    W = Path(a.work) / "vol"
    pats = sorted(n for n in core_ref if (W / "full" / f"training_{n}.npz").exists())
    fold = ta.folds({n: groups[n] for n in pats})
    qi = int(np.argmin(np.abs(Q_GRID - 0.5)))
    i10, i90 = int(np.argmin(np.abs(Q_GRID - 0.1))), int(np.argmin(np.abs(Q_GRID - 0.9)))
    med = {n: np.load(W / "full" / f"training_{n}.npz") for n in pats}
    lc = {n: np.log1p(med[n]["core_grid"][qi]) for n in pats}
    lh = {n: np.log1p(med[n]["hypo_grid"][qi]) for n in pats}
    choice = {}
    for k in range(5):
        tr = [n for n in pats if fold[n] != k]
        ec = np.median(np.abs(np.array([lc[n] for n in tr]) - np.log1p([core_ref[n] for n in tr])[:, None]), axis=0)
        eh = np.median(np.abs(np.array([lh[n] for n in tr]) - np.log1p([hypo_ref[n] for n in tr])[:, None]), axis=0)
        choice[k] = (int(np.argmin(ec)), int(np.argmin(eh)))
    out = {}
    for L in LENGTHS:
        rec = {}
        for n in pats:
            f = W / L / f"training_{n}.npz"
            if not f.exists():
                continue
            z, zf = np.load(f), np.load(W / "full" / f"training_{n}.npz")
            i, j = choice[fold[n]]
            core = float(z["core_grid"][qi][i])
            hyp = z["hypo_grid"][:, j].astype(float)
            lo, hi = np.log1p(min(hyp[i10], hyp[i90])), np.log1p(max(hyp[i10], hyp[i90]))
            ref = np.log1p(hypo_ref[n])
            cf, hf = float(zf["core_grid"][qi][i]), float(zf["hypo_grid"][qi][j])
            rec[n] = {"lo": lo, "hi": hi, "ref": ref, "score": max(lo - ref, ref - hi), "core": core,
                      "p": float(np.mean([ta.defuse3(core, max(core, float(h))) for h in hyp])),
                      "d_point": ta.defuse3(core, max(core, float(hyp[qi]))),
                      "d_full": ta.defuse3(cf, max(cf, hf)), "d_ref": dec_ref[n]}
        ns = sorted(rec)
        cov, wid, certain = [], [], {}
        for n in ns:
            others = [rec[m]["score"] for m in ns if fold[m] != fold[n]]
            qhat = float(np.quantile(others, 0.8 * (1 + 1 / len(others))))
            lo, hi = rec[n]["lo"] - qhat, rec[n]["hi"] + qhat
            if lo > hi:
                lo = hi = 0.5 * (lo + hi)
            cov.append(lo <= rec[n]["ref"] <= hi)
            wid.append(float(np.expm1(hi) - np.expm1(max(lo, 0.0))))
            c = rec[n]["core"]
            d_lo, d_hi = ta.defuse3(c, max(c, float(np.expm1(max(lo, 0.0))))), ta.defuse3(c, max(c, float(np.expm1(hi))))
            certain[n] = d_lo == d_hi
        flip = np.array([rec[n]["d_point"] != rec[n]["d_full"] for n in ns])
        agree = np.array([rec[n]["d_point"] == rec[n]["d_ref"] for n in ns])
        cert = np.array([certain[n] for n in ns])
        conf = np.array([abs(rec[n]["p"] - 0.5) for n in ns])
        order = np.argsort(-conf, kind="stable")
        rc = {}
        for frac in (0.5, 0.7, 0.9, 1.0):
            top = order[:max(1, int(round(frac * len(ns))))]
            rc[str(frac)] = {"agree": float(agree[top].mean()), "flip": float(flip[top].mean())}
        res = {"n": len(ns), "cqr_cover80": float(np.mean(cov)), "cqr_width80_median_ml": float(np.median(wid)),
               "raw_cover80": float(np.mean([rec[n]["lo"] <= rec[n]["ref"] <= rec[n]["hi"] for n in ns])),
               "certain_share": float(cert.mean()),
               "flip_certain": float(flip[cert].mean()) if cert.any() else None,
               "flip_uncertain": float(flip[~cert].mean()) if (~cert).any() else None,
               "agree_certain": float(agree[cert].mean()) if cert.any() else None,
               "agree_all": float(agree.mean()), "risk_coverage": rc}
        out[L] = res
        print(L.rjust(4), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "risk_coverage"}))
        print("     risk-coverage (agree | flip):", "  ".join(f"top {int(100 * float(k))}%: {v['agree']:.2f} | {v['flip']:.2f}"
                                                         for k, v in rc.items()), flush=True)
    (Path(a.work) / "score2.json").write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("calib", "maps", "volumes", "score", "score2"))
    p.add_argument("args", nargs="+")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    if a.stage == "calib":
        a.pairs, a.weights, a.work = a.args
        calib(a)
    elif a.stage == "maps":
        a.data_root, a.twin_out, a.weights, a.work = a.args
        maps(a)
    elif a.stage == "volumes":
        a.data_root, a.twin_out, a.work = a.args
        volumes(a)
    elif a.stage == "score2":
        a.results, a.work = a.args
        score2(a)
    else:
        a.results, a.work = a.args
        score(a)


if __name__ == "__main__":
    main()
