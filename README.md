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

## Results

Agreement with the vendor DEFUSE 3 decision (%), per cut length. **Forward** trains on ISLES'24
and is applied to ISLES'18 (94 slabs, always-NO 79.8 %); **reverse** trains on ISLES'18 and is
applied to all 145 ISLES'24 patients (always-YES 91.7 %). Subscripts are the seed SD. The twin is
a *target* in AmorTwin-2D and a *decoder* in AmorTwin.

| direction | cut | CNN | control | TTA | filt | AT-2D | **AT** |
|---|---|---|---|---|---|---|---|
| forward | 25 s | 88.7±0.9 | 88.7±1.3 | 92.2±2.2 | 91.5±1.1 | 93.4±0.4 | **94.3±0.5** |
| forward | 30 s | 87.9±0.9 | 89.0±1.8 | 92.2±1.6 | 92.3±1.8 | 94.3±1.3 | **96.5±0.5** |
| forward | 35 s | 86.2±1.9 | 88.3±2.6 | 91.5±1.8 | 91.9±1.0 | 94.3±0.5 | **94.3±0.5** |
| reverse | 25 s | 58.2±14.4 | 83.9±7.5 | 83.7±2.1 | 85.8±4.1 | 75.4±11.6 | **92.3±0.9** |
| reverse | 30 s | 70.6±7.8 | 85.6±4.8 | 87.8±1.4 | 89.1±2.0 | 77.7±10.5 | **91.7±0.4** |
| reverse | 35 s | 69.0±11.6 | 85.9±5.6 | 89.2±2.0 | 89.8±1.4 | 77.7±9.4 | **91.8±0.5** |

The same transport model, network and data give a 15-point spread in reverse depending only on
whether the model is a target or a layer. `TTA` and `filt` are given target data the twin-informed
arms never see, and still do not close the gap.

### What carries the transfer

Keeping the decoder and swapping only the trunk changes little; replacing the symbolic decoder by
a learned 8.1 k-parameter MLP of the same interface keeps the gain forward and loses it in reverse.

| direction | cut | residual | learned dec. | AT |
|---|---|---|---|---|
| forward | 25 s | 93.6±1.1 | 92.1±2.2 | 94.3±0.5 |
| forward | 30 s | 94.3±1.2 | 90.4±2.1 | 96.5±0.5 |
| forward | 35 s | 92.9±0.6 | 90.4±2.5 | 94.3±0.5 |
| reverse | 25 s | 88.3±1.9 | 76.6±10.7 | 92.3±0.9 |
| reverse | 30 s | 89.9±1.0 | 83.2±12.4 | 91.7±0.4 |
| reverse | 35 s | 88.7±4.2 | 80.9±12.4 | 91.8±0.5 |

`scripts/encoders/baselines.py` also provides an attention-gated trunk, reachable with
`--arch attunet`. It is not listed above; it tracks AT within about a point in either direction,
which is the same conclusion the residual arm supports.

Forward uses five seeds for the learned decoder and three elsewhere; reverse uses three, and ten
for AT and the control in the first table.

### Against published methods

Measured on the same 94 ISLES'18 slabs at the 30 s cut, on one RTX 3080 Ti.

| method | truncated | cross-centre | core | hypo vol. err | empty | agree. | time |
|---|---|---|---|---|---|---|---|
| SPPINN | no | no | yes | — | — | — | — |
| ReSPPINN | no | no | yes | — | — | — | — |
| PerfU-Net | no | yes | yes | — | — | — | — |
| Truncated-CTP | yes | no | no | — | — | — | — |
| HemoPIC (as AT-2D targets) | yes | yes | yes | — | — | 77.7 % | 1.1 s |
| EPPINN (our run) | yes | no | no | 1.9 ml | 5/94 | — | 150 s |
| **AmorTwin** | yes | yes | yes | **1.1 ml** | **0/94** | **91.7 %** | **1.1 s** |

Volume error is the median absolute hypoperfused-volume error; `agree.` is reverse-transfer
DEFUSE 3 agreement at 30 s. Dashes mark quantities a method does not produce: EPPINN emits no
probability and so yields no DEFUSE 3 decision, and the per-scan networks have no cross-centre
cohort to score. AmorTwin read out EPPINN's way (a hard threshold at 0.5 rather than the
probability weighting used throughout) gives 1.2 ml, still below EPPINN's 1.9 ml.

This is Table 3 of the paper, which lists only the rows with a measured number.

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
