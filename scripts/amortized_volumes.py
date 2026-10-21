"""E3b: core and hypoperfusion volume grids of the truncation-aware AmorTwin, in the format of
duofusion/scripts/osvd_volumes.py, so that duofusion/scripts/truncation_accuracy.py scores it like
the other estimators.

python scripts/amortized_volumes.py <data_root> <twin_out> <weights_dir> <results_dir> --yy 24
       --lengths full 25 30 35 40 45 [--name amortized] [--ensemble] [--sigma 3]

Per patient and scan length L (the CTP cut to its first L s as duofusion/scripts/truncate_ctp.py,
in memory), from the raw curves only:
  AIF      hemopic_plus.aif.select_aif on the truncated scan (as time_amortized.py)
  network  (point or probabilistic checkpoints; the posterior mean is used) the model of the patient's held-out fold (ISLES'24) or the mean of the five
           (--ensemble, external), inputs as export_pairs.py with the length of the cut scan;
           outputs ln rel. MTT, delay, ln rel. CBV and the twin Tmax of the full scan
  maps     CBF = CBV / MTT (relative), Tmax; then exactly as osvd_volumes.py: nan-aware Gaussian
           smoothing (--sigma mm), Tmax relative to the less delayed hemisphere, rCBF against
           the contralateral healthy (Tmax_rel < 2 s) median, volumes on TAU_GRID x DT_GRID.
The twin output folder is read only for the patient list and the ROI labels (the brain mask of
the other estimators). Writes <results_dir>/<name><yy>_<full|T L>/training_X.npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DF = ROOT / "external" / "duofusion"
for p_ in (DF, DF / "external" / "HemoPIC_plus", HERE):
    sys.path.insert(0, str(p_))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from amortized_model import TARGETS4, TMAX, AmorTwin, predict  # noqa: E402
from duofusion.data import load_subject  # noqa: E402
from duofusion.reference import affected_side, nan_gaussian, spacing_of  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import SEG4_REL, load_nifti  # noqa: E402
from hemopic_plus.npde.voxel import smooth_curves  # noqa: E402
from time_amortized import auto_aif  # noqa: E402

TAU_GRID = np.round(np.arange(0.10, 0.551, 0.05), 3)
DT_GRID = np.round(np.arange(0.0, 8.01, 0.5), 2)
LAB = "roi_labels_seg4_within_kmeans.nii.gz"


def grids_from_maps(s, brain, cbf, tmax, sigma):
    """osvd_volumes.py post-processing: smoothing, relative Tmax, contralateral rCBF, volume grids."""
    sp = spacing_of(s.affine)
    ok = brain & np.isfinite(cbf) & (cbf > 0) & np.isfinite(tmax)
    if sigma > 0:
        cbf, tmax = nan_gaussian(np.where(ok, cbf, np.nan), sigma, sp), nan_gaussian(np.where(ok, tmax, np.nan), sigma, sp)
    valid = brain & np.isfinite(cbf) & (cbf > 0) & np.isfinite(tmax)
    med = [np.median(tmax[valid & (s.hemi == h)]) if np.any(valid & (s.hemi == h)) else np.inf for h in (1, 2)]
    trel = tmax - (min(med) if np.isfinite(min(med)) else np.median(tmax[valid]))
    healthy = valid & (trel < 2.0)
    side = affected_side(s, valid & (trel > 4.0))
    contra = (s.hemi == 3 - side) if side else (s.hemi > 0)
    sel = healthy & contra if np.any(healthy & contra) else (healthy if np.any(healthy) else valid)
    r = cbf / float(np.median(cbf[sel]))
    cores = [valid & (r < t) for t in TAU_GRID]
    hypos = [valid & (trel > t) for t in DT_GRID]
    ml = s.voxel_ml
    return {"tau_grid": TAU_GRID, "dt_grid": DT_GRID,
            "core_grid": np.array([c.sum() for c in cores]) * ml,
            "hypo_grid": np.array([h.sum() for h in hypos]) * ml,
            "union_grid": np.array([[(c | h).sum() for h in hypos] for c in cores]) * ml}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("--yy", default="24")
    p.add_argument("--lengths", nargs="+", default=["full", "25", "30", "35", "40", "45"])
    p.add_argument("--name", default="amortized")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--config", type=Path, default=DF / "external" / "HemoPIC_plus" / "configs" / "isles24_full_aif.yaml")
    p.add_argument("--folds", type=Path, default=HERE / "learned_folds24.json")
    p.add_argument("--shard", default="0/1", help="k/n: process every n-th patient from the k-th (parallel CPU runs)")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(a.config, {"data_root": str(a.data_root)})
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    models = {}
    for k in sorted(set(folds.values())):
        ck = torch.load(a.weights / f"amortized_fold{k}.pt", map_location=dev)
        assert tuple(ck["targets"]) == TARGETS4, ck.get("targets")
        n_out = ck["state"]["head.2.weight"].shape[0]            # 4, or 8 for the probabilistic model (means first)
        m = AmorTwin(n_out=n_out).to(dev)
        m.load_state_dict(ck["state"])
        models[k] = (m.eval(), ck["mu"].cpu().numpy(), ck["sd"].cpu().numpy())
    outs = {L: a.results / f"{a.name}{a.yy}_{'full' if L == 'full' else 'T' + L}" for L in a.lengths}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    nums = sorted(int(d.name.split("_")[1]) for d in (a.twin_out / "isles24_full_aif").glob("training_*"))
    sk, sn = (int(v) for v in a.shard.split("/"))
    for n in nums[sk::sn]:
        if all((outs[L] / f"training_{n}.npz").exists() for L in a.lengths):
            continue
        fit_dir = a.twin_out / "isles24_full_aif" / f"training_{n}"
        if not (fit_dir / "run_info_core.json").exists():
            continue
        t0 = time.time()
        s = load_subject(a.data_root, n)
        ctc_full = s.ctc()
        seg4 = load_nifti(a.data_root / f"training_{n}" / SEG4_REL)[1].astype(np.int32)
        lab = np.rint(nib.load(str(fit_dir / LAB)).get_fdata()).astype(int)
        info = json.loads((fit_dir / "run_info_core.json").read_text())
        dt = float(info["dt_seconds"])
        use = list(models.values()) if (a.ensemble or n not in folds) else [models[folds[n]]]
        brain_all = (s.tissue > 0) & (lab > 0)
        rec = {}
        for L in a.lengths:
            keep = ctc_full.shape[3] if L == "full" else min(ctc_full.shape[3], int(np.floor(float(L) / dt)) + 1)
            ctc = ctc_full[..., :keep]
            fin = np.all(np.isfinite(ctc), axis=3)
            gmwm = np.isin(seg4, (1, 2)) & fin
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
            args = (curve, np.broadcast_to(ctx, (N, TMAX)), np.broadcast_to(aif, (N, TMAX)), np.full(N, T), seg4[gmwm])
            pred = np.mean([predict(m_, *args, device=dev)[:, :len(TARGETS4)] * sd_ + mu_ for m_, mu_, sd_ in use], axis=0)
            cbf, tmax = np.full(seg4.shape, np.nan), np.full(seg4.shape, np.nan)
            cbf[gmwm] = np.exp(pred[:, 2] - pred[:, 0])
            tmax[gmwm] = pred[:, 3]
            g = grids_from_maps(s, brain_all & gmwm, cbf, tmax, a.sigma)
            np.savez_compressed(outs[L] / f"training_{n}.npz", **g)
            rec[L] = round(float(g["core_grid"][4]), 1), round(float(g["hypo_grid"][4]), 1)
        print(json.dumps({"patient": n, "seconds": round(time.time() - t0, 1), "core30_hypo_rel2": rec}), flush=True)


if __name__ == "__main__":
    main()
