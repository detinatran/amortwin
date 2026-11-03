# Baselines

Two published methods, run on our cohorts so that the comparison uses one reference and one metric.

## EPPINN

`to_eppinn.py` converts our `.npz` cases into the per-case NIfTI layout the released EPPINN code
expects (CTP frames, AIF, brain mask). The model itself is the authors' code, unmodified.

EPPINN solves an optimisation per scan, so it has no train-here-test-there quantity: it cannot
enter the transfer table. It also emits no infarct core -- its maps carry no contralateral CBF
reference -- so only the hypoperfused volume is comparable. Measured on 94 ISLES'18 slabs at the
30 s cut: median absolute error 1.9 ml, 5 of 94 cases returning an empty map, 150 s per scan.

## 4D mJ-Net

`conv4d.py` and `mjnet4d.py` port the architecture of "CT Perfusion is All We Need" from
TensorFlow, whose stack needs CUDA 10.1 and does not run on current cards. `train_mjnet.py` and
`infer_mjnet.py` drive it on our data.

**These numbers are not reported in the paper, and the port should not be read as an evaluation
of the original method.** Conv4D was checked numerically against the TF reference and agrees to
9.5e-07, but the run did not learn: the loss settled at 0.688, which is ln 2, so the network
emits about 0.5 everywhere. Three faults in this port are enough to explain that on their own,
and none of them is a property of the published method:

1. `limchan=2` narrows all three 4D blocks to 4 channels, against 16/32 in the reference.
2. `forward` averages over the time axis (`x.mean(dim=3)`) after the strided blocks, which
   discards the curve shape that perfusion lives in.
3. `train_mjnet.py` collapses the 3D label with `tgt.amax(1)`, so one patient is one training
   sample and the target is a max projection through every slice.

Fixing these means rewriting the 4D branch and the label pipeline, then retraining; at ~400 s per
epoch that is days of compute, which is why the arm was stopped rather than tuned.
