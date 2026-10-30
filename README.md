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

### 2D models

The voxelwise model above is amortized over single curves. The models below work on a whole slice
cut at `L` seconds and are the ones evaluated across centres.

| file | purpose |
|---|---|
| `scripts/cnn_baseline.py` | the 2D U-Net trunk (48/96/192, bottleneck 384) and the curves-only baseline |
| `scripts/amortwin2d.py` | AmorTwin-2D: the twin's fitted parameters as regression targets, plus inference and evaluation shared by every 2D arm |
| `scripts/at2d_sae.py` | **AmorTwin-S**: the twin's indicator-dilution model as a fixed, weight-free decoder inside the network; the encoder emits per-pixel (ln MTT, delay, ln CBV) and the decoder rebuilds the observed part of each curve |
| `scripts/at2d_tta.py` | the decoder's rebuild/AIF-shift helpers, and test-time adaptation baselines |
| `scripts/mvols_dir.py` | probability-weighted volumes `V = sum_v p(v) w(v)` from a prediction directory |

## Dependencies

* Python 3.11+, PyTorch, NumPy, SciPy, nibabel (see `requirements.txt`).
* The physics twin (extended) in `external/twin`, and the truncation benchmark and data
  loaders in `external/benchmark`; both are imported by path from the scripts.
* Data: ISLES'24, ISLES'18 and UniToBrain (public; not redistributed here).

## Typical run

```bash
python scripts/export_pairs.py <ISLES2024_root> <twin_out> pairs.npz --folds scripts/learned_folds24.json
python scripts/amortized_train.py pairs.npz weights/ --trunc-min 25 --prob --augment
python scripts/prob_amortwin.py calib pairs.npz weights/ work/
python scripts/eval_prob_external.py <external_pairs.npz> weights/ work/calib.json out.json
```

AmorTwin-S, trained on one centre and applied to another without retraining:

```bash
# train: the symbolic decoder is self-supervised, so --twin-weight 0 uses no twin fit at all
python scripts/at2d_sae.py train <cnn_dir> <x_dir> scripts/learned_folds24.json work/ \
    --masks --fold 0 --epochs 60 --rec-weight 1 --twin-weight 0 --seed 0

# apply to a second centre and read out the volumes
python scripts/at2d_sae.py infer <cnn_dir2> <x_dir2> - work/ --masks --external pred2
python scripts/mvols_dir.py work/pred2 work/volumes2.json
```
