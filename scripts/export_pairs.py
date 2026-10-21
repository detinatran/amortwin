"""Training pairs for AmorTwin: voxel curve + AIF -> twin voxel parameters.

python scripts/export_pairs.py <ISLES2024_Training_s3> <twin_out_s3> <out.npz> --folds scripts/learned_folds24.json

Per patient (the smoothed ISLES'24 copy of the paper and its twin fits), up to --per-patient
tissue voxels with a voxel fit, a third inside the final infarct when possible:
  curve   the voxel concentration curve, temporally smoothed as in the voxel fit
          (hemopic_plus.npde.voxel.smooth_curves), first TMAX frames, zero-padded
  aif     the twin automatic AIF of the patient (normalised), same length
  ctx     the patient's median smoothed tissue curve over all fitted tissue voxels (context:
          the CBV and MTT targets are relative to the brain median)
  length  number of measured frames
  tissue  HemoPIC tissue class (1 grey matter, 2 white matter)
  targets ln(MTT / brain median MTT), voxel delay (s), ln(CBV / brain median CBV), with the
          twin voxel MTT, voxel delay and indicator-dilution CBV (the paper's voxel maps), and
          the twin Tmax (s) = delay + time to peak of exp(-t/MTT) fed through the exponential
          dispersion of the voxel's ROI (as duofusion/scripts/duofusion_volumes.py);
          with --physics also ln MTT (s) and ln of the ROI dispersion beta (s), so that a model can
          compose Tmax = delay + residue peak(beta, MTT) from its own outputs (amortized_model.TARGETS6)
  roi     with --roi-context: the median smoothed curve of every ROI of the twin's segmentation
          (roi_labels_seg4_within_kmeans, no fit needed) over its fitted tissue voxels (roi_curve),
          and the index of the voxel's ROI curve (roi_index)
  lesion  inside the final infarct (for sampling and analysis only, never an input)
plus patient id and fold. Writes one compressed npz.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "HemoPIC_plus"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, squeeze_ctc  # noqa: E402
from hemopic_plus.npde.voxel import smooth_curves  # noqa: E402
from hemopic_plus.uq.data import load_subject as load_fit  # noqa: E402

TMAX = 70
LAB = "roi_labels_seg4_within_kmeans.nii.gz"


def residue_peak(beta, mtt):
    """Time to peak of exp(-t/MTT) convolved with the dispersion kernel exp(-t/beta)/beta."""
    beta, mtt = np.maximum(beta, 1e-3), np.maximum(mtt, 1e-3)
    close = np.abs(mtt - beta) < 1e-3 * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta * mtt * np.log(mtt / beta) / (mtt - beta)
    return np.where(close, beta, t)


def work(args):
    data_root, duo, n, per, seed, physics, roi_ctx = args
    d = data_root / f"training_{n}"
    fit_dir = duo / "isles24_full_aif" / f"training_{n}"
    vd = duo / "isles24_full_aif__vox_tank" / f"training_{n}"
    cbv_f = duo / "isles24_full_aif_idvox_osvd_nc" / f"training_{n}" / "HemoPIC_CBV.nii.gz"
    if not ((vd / "VOX_MTT.nii.gz").exists() and cbv_f.exists() and (fit_dir / "fit_curves.npz").exists()):
        return None
    fit = load_fit(fit_dir)
    cfg = fit.cfg
    lab = np.rint(nib.load(str(fit_dir / LAB)).get_fdata()).astype(int)
    disp = np.full(lab.max() + 1, np.nan)
    for i, rid in enumerate(fit.roi_ids):
        disp[int(rid)] = float(np.exp(fit.theta_fit[i].numpy()[6]))
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
        return None
    med_mtt, med_cbv = float(np.median(mtt[ok])), float(np.median(cbv[ok]))
    rng = np.random.default_rng(seed + n)
    les, rest = np.argwhere(ok & ot), np.argwhere(ok & ~ot)
    k_les = min(len(les), per // 3)
    pick = np.concatenate([les[rng.choice(len(les), k_les, replace=False)] if k_les else np.zeros((0, 3), int),
                           rest[rng.choice(len(rest), min(len(rest), per - k_les), replace=False)]])
    idx = tuple(pick.T)
    c = smooth_curves(ctc[idx], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
    allc = smooth_curves(ctc[ok], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
    ctx = np.zeros(TMAX, np.float32)
    ctx[:min(allc.shape[1], TMAX)] = np.median(allc, axis=0)[:TMAX]
    T = min(c.shape[1], TMAX)
    curve = np.zeros((len(pick), TMAX), np.float16)
    curve[:, :T] = c[:, :T]
    a = np.load(fit_dir / "fit_curves.npz")["cart_norm"]
    aif = np.zeros(TMAX, np.float16)
    aif[:min(a.size, TMAX)] = a[:TMAX]
    tmax = delay[idx] + residue_peak(beta[idx], mtt[idx])
    cols = [np.log(mtt[idx] / med_mtt), delay[idx], np.log(cbv[idx] / med_cbv), tmax]
    if physics:
        cols += [np.log(mtt[idx]), np.log(beta[idx])]
    tgt = np.stack(cols, 1).astype(np.float32)
    extra = {}
    if roi_ctx:
        labs = lab[ok]
        ids = np.unique(labs)
        rc = np.zeros((len(ids), TMAX), np.float16)
        for j, rid in enumerate(ids):
            rc[j, :min(allc.shape[1], TMAX)] = np.median(allc[labs == rid], axis=0)[:TMAX]
        extra = {"roi_curve": rc, "roi_local": np.searchsorted(ids, lab[idx]).astype(np.int32)}
    return {**extra, "pid": n, "curve": curve, "aif": aif, "ctx": ctx, "length": T, "tissue": seg[idx].astype(np.int8),
            "target": tgt, "lesion": ot[idx]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--per-patient", type=int, default=20000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--physics", action="store_true", help="add the ln MTT and ln dispersion targets")
    p.add_argument("--roi-context", action="store_true", help="add the median curve of each voxel's ROI")
    a = p.parse_args()
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = [r for r in ex.map(work, [(a.data_root, a.twin_out, n, a.per_patient, 0, a.physics, a.roi_context) for n in sorted(folds)]) if r]
    rep = lambda r, k: np.repeat(r[k], len(r["target"]))  # noqa: E731
    out = {"curve": np.concatenate([r["curve"] for r in res]),
           "aif": np.stack([r["aif"] for r in res]),
           "ctx": np.stack([r["ctx"] for r in res]),
           "patient_index": np.concatenate([np.full(len(r["target"]), i, np.int32) for i, r in enumerate(res)]),
           "length": np.concatenate([np.full(len(r["target"]), r["length"], np.int16) for r in res]),
           "tissue": np.concatenate([r["tissue"] for r in res]),
           "target": np.concatenate([r["target"] for r in res]),
           "lesion": np.concatenate([r["lesion"] for r in res]),
           "pid": np.concatenate([np.full(len(r["target"]), r["pid"], np.int16) for r in res]),
           "fold": np.concatenate([np.full(len(r["target"]), folds[r["pid"]], np.int8) for r in res])}
    del rep
    if a.roi_context:
        off = np.cumsum([0] + [len(r["roi_curve"]) for r in res[:-1]])
        out["roi_curve"] = np.concatenate([r["roi_curve"] for r in res])
        out["roi_index"] = np.concatenate([r["roi_local"] + o for r, o in zip(res, off)]).astype(np.int32)
    np.savez_compressed(a.out, **out)
    print(len(res), "patients,", len(out["target"]), "voxels ->", a.out, f"{a.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
