"""Lesion AUC of the AmorTwin maps against the twin voxel fit and the vendor maps (ISLES'24).

python scripts/eval_amortized.py <ISLES2024_Training_s3> <twin_out_s3> <weights_dir> <out_dir>
       [--folds scripts/learned_folds24.json] [--device auto]
python scripts/eval_amortized.py <ISLES2018_Training> <twin_out2018> <weights_dir> <out_dir> --ensemble


Per patient (held-out: the model of the patient's own fold, trained without it; with --ensemble,
for an external cohort, the mean of the five models), the network
of amortized_model.py runs on every grey/white-matter voxel with the inputs of export_pairs.py
(smoothed voxel curve, twin automatic AIF, median tissue curve, length, tissue class) and
gives ln(MTT/median), delay and ln(CBV/median); CBF = CBV / MTT (relative maps; the AUC is
scale-free). Scored exactly as scripts/eval_external.py of HemoPIC_plus: lesion ROC AUC of CBF
(low), CBV (low) and MTT (high) relative to the brain median, against the follow-up infarct,
  brain   seg4 > 0 (as the paper; the amortized maps exist on grey/white matter only)
  gmwm    grey/white matter with a finite curve, the same voxels for every method
for amortized, twin-vox (ID-CBV, voxel MTT, CBF = 60 CBV / MTT) and the vendor maps.
Writes per_patient.csv and summary.json (means, paired differences, Wilcoxon p, timing).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for hp in (ROOT / "external" / "HemoPIC_plus", ROOT / "external" / "duofusion" / "external" / "HemoPIC_plus"):
    if hp.exists():
        sys.path.insert(0, str(hp))
        break
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from amortized_model import TMAX, AmorTwin, predict  # noqa: E402
from hemopic_plus.evaluation import MAPS, _lesion_scores, roc_auc  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, squeeze_ctc  # noqa: E402
from hemopic_plus.npde.voxel import smooth_curves  # noqa: E402
from hemopic_plus.uq.data import load_subject as load_fit  # noqa: E402


def auc(vol, mask, ot, direction):
    valid, score = _lesion_scores(vol, mask, direction)
    return roc_auc(ot[valid], score) if score is not None else np.nan


def gpu_ok(busy=90.0):
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return torch.cuda.is_available() and float(out[0]) < busy
    except Exception:  # noqa: BLE001
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--folds", type=Path, default=Path(__file__).resolve().parent / "learned_folds24.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--ensemble", action="store_true", help="external cohort: mean of the five fold models")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = ("cuda" if gpu_ok() else "cpu") if a.device == "auto" else a.device
    torch.set_num_threads(16)
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    fold_ids = sorted(set(folds.values()))
    if a.ensemble:
        folds = {int(d.name.split("_")[1]): "all" for d in (a.twin_out / "isles24_full_aif").glob("training_*")}
    models = {}
    for k in fold_ids:
        ck = torch.load(a.weights / f"amortized_fold{k}.pt", map_location=dev)
        m = AmorTwin().to(dev)
        m.load_state_dict(ck["state"])
        models[k] = (m.eval(), ck["mu"].cpu().numpy(), ck["sd"].cpu().numpy())
    print("device", dev, "models", sorted(models), flush=True)

    rows = []
    for n in sorted(folds):
        sub = f"training_{n}"
        base = a.data_root / sub
        fit_dir = a.twin_out / "isles24_full_aif" / sub
        vd = a.twin_out / "isles24_full_aif__vox_tank" / sub
        cbv_f = a.twin_out / "isles24_full_aif_idvox_osvd_nc" / sub / "HemoPIC_CBV.nii.gz"
        if not ((fit_dir / "fit_curves.npz").exists() and (vd / "VOX_MTT.nii.gz").exists() and cbv_f.exists()):
            continue
        seg = load_nifti(base / SEG4_REL)[1].astype(int)
        ot = load_nifti(find_nii(base, "OT"))[1] > 0
        brain = seg > 0
        if not (ot & brain).any():
            continue
        cfg = load_fit(fit_dir).cfg
        ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
        gmwm = np.isin(seg, (1, 2)) & np.all(np.isfinite(ctc), axis=3)
        t0 = time.time()
        c = smooth_curves(ctc[gmwm], cfg.smooth_sigma_frames, cfg.baseline_frames, cfg.clip_nonneg)
        T = min(c.shape[1], TMAX)
        curve = np.zeros((len(c), TMAX), np.float32)
        curve[:, :T] = c[:, :T]
        ctx = np.zeros(TMAX, np.float32)
        ctx[:T] = np.median(c, axis=0)[:T]
        art = np.load(fit_dir / "fit_curves.npz")["cart_norm"]
        aif = np.zeros(TMAX, np.float32)
        aif[:min(art.size, TMAX)] = art[:TMAX]
        N = len(curve)
        args = (curve, np.broadcast_to(ctx, (N, TMAX)), np.broadcast_to(aif, (N, TMAX)), np.full(N, T), seg[gmwm])
        use = list(models.values()) if folds[n] == "all" else [models[folds[n]]]
        pred = np.mean([predict(m_, *args, device=dev) * sd_ + mu_ for m_, mu_, sd_ in use], axis=0)
        if dev == "cuda":
            torch.cuda.synchronize()
        secs = time.time() - t0
        am = {k: np.full(seg.shape, np.nan) for k in ("MTT", "CBV", "delay")}
        am["MTT"][gmwm], am["delay"][gmwm], am["CBV"][gmwm] = np.exp(pred[:, 0]), pred[:, 1], np.exp(pred[:, 2])
        am["CBF"] = am["CBV"] / am["MTT"]
        mtt = nib.load(str(vd / "VOX_MTT.nii.gz")).get_fdata()
        cbv = nib.load(str(cbv_f)).get_fdata()
        duo = {"CBV": cbv, "MTT": mtt, "CBF": np.where(np.isfinite(mtt) & (mtt > 0), 60.0 * cbv / np.maximum(mtt, 1e-6), np.nan)}
        ven = {m: load_nifti(find_nii(base, f))[1] for m, f in (("CBF", "MR_rCBF"), ("CBV", "MR_rCBV"), ("MTT", "MR_MTT"))}
        r = {"patient": n, "fold": folds[n], "voxels": int(N), "seconds": round(secs, 2)}
        for name, vols in (("amortized", am), ("twin-vox", duo), ("vendor", ven)):
            for mask_name, mask in (("brain", brain), ("gmwm", gmwm)):
                vals = []
                for m, direction in MAPS:
                    v = auc(vols[m], mask, ot, direction)
                    r[f"{name} {mask_name} {m}"] = v
                    vals.append(v)
                r[f"{name} {mask_name} mean"] = float(np.mean(vals))
        both = gmwm & np.isfinite(duo["MTT"]) & (duo["MTT"] > 0) & np.isfinite(am["MTT"])
        for m in ("CBF", "CBV", "MTT"):
            x, y = np.log(np.maximum(am[m][both], 1e-9)), np.log(np.maximum(duo[m][both], 1e-9))
            ok = np.isfinite(x) & np.isfinite(y)
            r[f"r_log {m}"] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                          if k in ("patient", "voxels", "seconds") or k.endswith("brain mean")}), flush=True)

    keys = [k for k in rows[0] if k not in ("patient", "fold")]
    with open(a.out / "per_patient.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["patient", "fold", *keys])
        for r in rows:
            w.writerow([r["patient"], r["fold"], *[r[k] for k in keys]])

    def paired(x, y):
        d = np.array([r[x] - r[y] for r in rows])
        d = d[np.isfinite(d)]
        return {"diff": float(d.mean()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, "n": int(d.size),
                "wins": int((d > 0).sum())}

    S = {"n": len(rows), "device": dev, "mean": {k: float(np.nanmean([r[k] for r in rows])) for k in keys},
         "paired": {}}
    for mask in ("brain", "gmwm"):
        for ref in ("twin-vox", "vendor"):
            for m in ("CBF", "CBV", "MTT", "mean"):
                S["paired"][f"amortized - {ref} {mask} {m}"] = paired(f"amortized {mask} {m}", f"{ref} {mask} {m}")
        for m in ("CBF", "CBV", "MTT", "mean"):
            S["paired"][f"twin-vox - vendor {mask} {m}"] = paired(f"twin-vox {mask} {m}", f"vendor {mask} {m}")
    (a.out / "summary.json").write_text(json.dumps(S, indent=1), encoding="utf8")
    print(f"\nlesion AUC, {len(rows)} patients ({dev}; median {np.median([r['seconds'] for r in rows]):.2f} s per patient)")
    for mask in ("brain", "gmwm"):
        for name in ("amortized", "twin-vox", "vendor"):
            print(f"  {mask:5s} {name:11s} " + "  ".join(f"{m} {S['mean'][f'{name} {mask} {m}']:.3f}"
                                                       for m in ("CBF", "CBV", "MTT", "mean")))
    print("  voxelwise r (log, amortized vs twin-vox): " + "  ".join(f"{m} {S['mean'][f'r_log {m}']:.3f}" for m in ("CBF", "CBV", "MTT")))
    for k, v in S["paired"].items():
        if k.endswith("mean"):
            print(f"  {k:40s} {v['diff']:+.4f}  p={v['p']:.2g}  wins {v['wins']}/{v['n']}")


if __name__ == "__main__":
    main()
