# AmorTwin

AmorTwin is a spatial network regularized by a fixed symbolic transport decoder, for lesion
estimation on truncated cerebral CT perfusion. From a CTP slice observed up to any cut length, a
shared U-Net encoder predicts core and hypoperfusion probabilities together with per-pixel
transit-time, arrival-delay and blood-volume coordinates; a weight-free decoder, the reduced
indicator-dilution model of the twin, rebuilds the observed part of each curve to constrain
training. A Cramer-Rao analysis quantifies which twin parameters truncation leaves identifiable,
and only those enter the decoder.

The model is trained on one centre and applied to another without target-cohort adaptation, and
is scored by its agreement with the vendor DEFUSE 3 thrombectomy decision.

The numbers below are the ones reported in the accompanying paper, currently under review.

```bash
python scripts/at2d_sae.py train <cnn_dir> <x_dir> scripts/learned_folds24.json work/ \
    --masks --fold 0 --epochs 60 --twin-weight 0 --arch calnet
```

## Reproducing the numbers

The scorers read the per-patient volumes an inference run writes (`mvols*.json`, or `thr24.json`
from `mvols_thr.py`) and apply the DEFUSE 3 rule. Volumes are always probability-weighted,
`V = sum_v p(v) w(v)`, never thresholded. Point `AMORTWIN_ROOT` at the directory holding the arm
folders, `cnn18/`, `cnn24/` and `refs*.json`:

```bash
export AMORTWIN_ROOT=/path/to/work
python scripts/scoring/agree18_json.py saeF_s0 saeF_s1 saeF_s2   # forward, ISLES'18
python scripts/scoring/agree24_json.py rsaeF_s0 rsaeF_s1 rsaeF_s2  # reverse, all 145
python scripts/scoring/agree24_thr.py rev_att_s0 rev_att_s1 rev_att_s2  # arms written by mvols_thr.py
python scripts/scoring/cmp_vol.py                                 # EPPINN vs AT-S volumes
```

## Dependencies

* Python 3.11+, PyTorch, NumPy, SciPy, nibabel (see `requirements.txt`).
* The physics twin (extended) in `external/twin`, and the truncation benchmark and data
  loaders in `external/benchmark`; both are imported by path from the scripts.
* Data: ISLES'24, ISLES'18 and UniToBrain (public; not redistributed here).

## Typical run

The voxelwise pipeline below reads the twin fits through the internal packages that produced
them, so `export_pairs.py`, `amortized_train.py`, `prob_amortwin.py` and the other scripts in
the first table import modules that are not part of this repository. The 2D models, the
scorers and the baselines have no such dependency and run on the released code alone.

```bash
python scripts/export_pairs.py <ISLES2024_root> <twin_out> pairs.npz --folds scripts/learned_folds24.json
python scripts/amortized_train.py pairs.npz weights/ --trunc-min 25 --prob --augment
python scripts/prob_amortwin.py calib pairs.npz weights/ work/
python scripts/eval_prob_external.py <external_pairs.npz> weights/ work/calib.json out.json
```

AmorTwin, trained on one centre and applied to another without retraining:

```bash
# train: the symbolic decoder is self-supervised, so --twin-weight 0 uses no twin fit at all
python scripts/at2d_sae.py train <cnn_dir> <x_dir> scripts/learned_folds24.json work/ \
    --masks --fold 0 --epochs 60 --rec-weight 1 --twin-weight 0 --seed 0

# apply to a second centre and read out the volumes
python scripts/at2d_sae.py infer <cnn_dir2> <x_dir2> - work/ --masks --external pred2
python scripts/mvols_dir.py work/pred2 work/volumes2.json
```
