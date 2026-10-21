"""E2: AmorTwin without the twin fit, and its run time (compare scripts/time_pipeline.py
of HemoPIC_plus, which times the regional + voxel twin fit on one CPU thread).

python scripts/time_amortized.py <data_root> <twin_out> <weights_dir> <out.json> --config <isles24_full_aif.yaml>
       --patients 1 2 ... [--ensemble]

Per patient, from the raw files only (the twin fit is read only to check the AIF):
  aif      hemopic_plus.aif.select_aif with the config of the twin run and its seed
           (cfg.seed + patient), normalised to its maximum, falling back to the mean curve of the
           bolus voxels as hemopic_plus.pipeline does; compared with the twin's cart_norm
  smooth   temporal smoothing of every grey/white-matter curve (as export_pairs.py)
  predict  the network (patient's fold model, or the mean of the five with --ensemble), on one
           CPU thread and, when a GPU is free (< 90 %), on the GPU
Loading the NIfTI files is timed separately. Writes per-patient times and the AIF check.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for hp in (ROOT / "external" / "HemoPIC_plus", ROOT / "external" / "duofusion" / "external" / "HemoPIC_plus"):
    if hp.exists():
        sys.path.insert(0, str(hp))
        break
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from amortized_model import TMAX, AmorTwin, predict  # noqa: E402
from hemopic_plus.aif import select_aif  # noqa: E402
from hemopic_plus.config import load_config  # noqa: E402
from hemopic_plus.io import SEG4_REL, find_nii, load_nifti, squeeze_ctc  # noqa: E402
from hemopic_plus.npde.voxel import smooth_curves  # noqa: E402
from hemopic_plus.signal import choose_peak_threshold, compute_peak_map, smooth_curve_gaussian  # noqa: E402


def gpu_free(busy=90.0):
    try:
        u = float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.split()[0])
        return torch.cuda.is_available() and u < busy
    except Exception:  # noqa: BLE001
        return False


def auto_aif(ctc, seg4, cfg, num, dt):
    """Normalised arterial input exactly as hemopic_plus.pipeline (auto_aif, else bolus-voxel mean)."""
    t = np.arange(ctc.shape[3], dtype=np.float64) * dt
    show = (seg4 > 0) & np.all(np.isfinite(ctc), axis=3)
    peak3 = compute_peak_map(ctc, baseline_frames=cfg.baseline_frames)
    fit_mask = show & (peak3 > float(choose_peak_threshold(peak3, show)))
    if int(fit_mask.sum()) < 5000:
        fit_mask = show
    cart = smooth_curve_gaussian(ctc[fit_mask].mean(axis=0).astype(np.float64), sigma_frames=cfg.smooth_sigma_frames,
                                 baseline_frames=cfg.baseline_frames, clip_nonneg=cfg.clip_nonneg)
    res = select_aif(ctc, show, t, cfg, seed=int(cfg.seed) + int(num)) if cfg.arterial_input == "auto_aif" else None
    if res is not None:
        cart = res["aif"].astype(np.float64)
    return cart / max(float(cart.max()), 1e-12), res is not None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_root", type=Path)
    p.add_argument("twin_out", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--patients", type=int, nargs="+", required=True)
    p.add_argument("--folds", type=Path, default=Path(__file__).resolve().parent / "learned_folds24.json")
    p.add_argument("--ensemble", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, {"data_root": str(a.data_root)})
    folds = {int(k): v for k, v in json.loads(a.folds.read_text()).items()}
    ck = {k: torch.load(a.weights / f"amortized_fold{k}.pt", map_location="cpu") for k in sorted(set(folds.values()))}

    def models(dev):
        out = {}
        for k, c in ck.items():
            m = AmorTwin()
            m.load_state_dict(c["state"])
            out[k] = (m.to(dev).eval(), c["mu"].cpu().numpy(), c["sd"].cpu().numpy())
        return out
    cpu_models = models("cpu")
    gpu = "cuda" if gpu_free() else None
    gpu_models = models(gpu) if gpu else None
    rows = []
    for n in a.patients:
        sub = f"training_{n}"
        base = a.data_root / sub
        t0 = time.perf_counter()
        seg4 = load_nifti(base / SEG4_REL)[1].astype(np.int32)
        ctc = squeeze_ctc(load_nifti(find_nii(base, "CTC_from_MR_4DPWI"))[1])
        t_load = time.perf_counter() - t0
        info = json.loads((a.twin_out / "isles24_full_aif" / sub / "run_info_core.json").read_text())
        dt = float(info["dt_seconds"])

        torch.set_num_threads(1)
        t0 = time.perf_counter()
        aif_full, auto = auto_aif(ctc, seg4, cfg, n, dt)
        t_aif = time.perf_counter() - t0
        twin_aif = np.load(a.twin_out / "isles24_full_aif" / sub / "fit_curves.npz")["cart_norm"]

        t0 = time.perf_counter()
        gmwm = np.isin(seg4, (1, 2)) & np.all(np.isfinite(ctc), axis=3)
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
        t_smooth = time.perf_counter() - t0

        keys = list(cpu_models) if a.ensemble or n not in folds else [folds[n]]
        t0 = time.perf_counter()
        pred_cpu = np.mean([predict(cpu_models[k][0], *args, device="cpu") * cpu_models[k][2] + cpu_models[k][1]
                            for k in keys], axis=0)
        t_cpu = time.perf_counter() - t0
        t_gpu = None
        if gpu_models:
            torch.set_num_threads(4)
            predict(gpu_models[keys[0]][0], *[x[:1000] for x in args], device=gpu)      # warm-up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_gpu = np.mean([predict(gpu_models[k][0], *args, device=gpu) * gpu_models[k][2] + gpu_models[k][1]
                                for k in keys], axis=0)
            torch.cuda.synchronize()
            t_gpu = time.perf_counter() - t0
            assert np.allclose(pred_gpu, pred_cpu, atol=1e-2), "CPU and GPU predictions differ"
        L = min(len(twin_aif), len(aif_full))
        r = {"patient": n, "voxels": int(N), "frames": int(ctc.shape[3]), "auto_aif": bool(auto),
             "aif_max_abs_diff_vs_twin": float(np.abs(aif_full[:L] - twin_aif[:L]).max()),
             "load_s": round(t_load, 3), "aif_s": round(t_aif, 3), "smooth_s": round(t_smooth, 3),
             "predict_cpu1_s": round(t_cpu, 3), "predict_gpu_s": None if t_gpu is None else round(t_gpu, 3),
             "total_cpu1_s": round(t_aif + t_smooth + t_cpu, 3),
             "total_gpu_s": None if t_gpu is None else round(t_aif + t_smooth + t_gpu, 3),
             "models": len(keys)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    med = {k: float(np.median([r[k] for r in rows if r[k] is not None]))
           for k in ("load_s", "aif_s", "smooth_s", "predict_cpu1_s", "predict_gpu_s", "total_cpu1_s", "total_gpu_s",
                     "aif_max_abs_diff_vs_twin") if any(r[k] is not None for r in rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"gpu": gpu, "median": med, "patients": rows}, indent=1))
    print("median", json.dumps(med))


if __name__ == "__main__":
    main()
