# AmorTwin

AmorTwin is an amortized posterior over the voxel parameters of a physics-based digital twin of
cerebral CT perfusion. From a voxel concentration curve cut at any scan length, a dilated 1D CNN
returns a Gaussian posterior over the twin's relative MTT, delay, relative CBV and Tmax. Combined
with the twin's hypoperfusion and normalised conformal calibration of the volume posterior, it
flags which truncated scans give a reliable thrombectomy (DEFUSE 3) decision.

The code is being released step by step together with the paper (in preparation).

## Layout

| file | purpose |
|---|---|
| `scripts/amortized_model.py` | network, input builder (`make_inputs`), target sets |
| `scripts/export_pairs.py` | training pairs from the twin fits of a cohort (curves, AIF, context, targets) |
| `scripts/amortized_train.py` | training per patient fold: random truncation, Gaussian NLL, acquisition augmentation |
| `scripts/prob_amortwin.py` | conformal calibration, posterior maps, volume quantiles, decision scoring |
| `scripts/eval_prob_external.py` | agreement and calibration on an external cohort without retraining |
| `scripts/robustness.py` | acquisition perturbations (frame rate, noise, late start, AIF shift) |
| `scripts/phantom.py` | phantom with known parameters; conformal calibration against the truth |
| `scripts/amortized_volumes.py` | core and hypoperfusion volume grids under truncation |
| `scripts/eval_amortized.py` | lesion AUC of the maps |
| `scripts/time_amortized.py` | run-time measurement |
| `scripts/learned_folds24.json` | patient folds of ISLES'24 used for all cross-fitting |

## Dependencies

* Python 3.11+, PyTorch, NumPy, SciPy, nibabel (see `requirements.txt`).
* The physics twin (HemoPIC, extended) in `external/HemoPIC_plus`, and the
  DuoFusion code (truncation benchmark, data loaders) in `external/duofusion`; both are imported by
  path from the scripts.
* Data: ISLES'24, ISLES'18 and UniToBrain (public; not redistributed here).

## Typical run

```bash
python scripts/export_pairs.py <ISLES2024_root> <twin_out> pairs.npz --folds scripts/learned_folds24.json
python scripts/amortized_train.py pairs.npz weights/ --trunc-min 25 --prob --augment
python scripts/prob_amortwin.py calib pairs.npz weights/ work/
python scripts/eval_prob_external.py <external_pairs.npz> weights/ work/calib.json out.json
```
