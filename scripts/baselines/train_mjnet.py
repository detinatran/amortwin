"""Train the ported 4D mJ-Net on our ISLES'24 data and infer to ISLES'18.

This is the architecture from "CT Perfusion is All We Need" (4d-mJ-Net), ported from
TensorFlow because its original stack needs CUDA 10.1 and cannot run on current cards.
Conv4D was checked numerically against the TF reference (9.5e-07); the decoder is rebuilt
from the reference's block descriptions rather than copied line by line, which is stated in
the paper if these numbers are used.

Input is the raw `ctc` array as (1, z, t, h, w) -- no pickle pipeline, unlike the reference.
Targets are the vendor core/hypoperfusion masks, the same supervision every other arm gets.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from mjnet4d import MJNet4D


def load_case(cnn_dir: Path, name: str, cut: int | None):
    z = np.load(cnn_dir / name)
    ctc = np.asarray(z["ctc"], dtype=np.float32)           # (z, t, h, w)
    if cut:
        ctc = ctc[:, :cut]
    core = np.asarray(z["core"], dtype=np.float32)          # (z, h, w)
    hypo = np.asarray(z["hypo"], dtype=np.float32)
    valid = np.asarray(z["valid"], dtype=np.float32)
    return ctc, np.stack([core, hypo]), valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cnn_dir"); p.add_argument("folds"); p.add_argument("work")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--cut", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    cnn = Path(a.cnn_dir); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)

    folds = {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    names = sorted(f.name for f in cnn.glob("*.npz"))
    def pid(n): return int(n.split("_")[1].split(".")[0])
    train = [n for n in names if folds.get(pid(n), -1) != a.fold]
    test = [n for n in names if folds.get(pid(n), -1) == a.fold]
    print(f"train {len(train)}  test {len(test)}", flush=True)

    c0, _, _ = load_case(cnn, train[0], a.cut)
    model = MJNet4D(z=c0.shape[0], t=c0.shape[1], out_ch=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; t0 = time.time()
        for n in np.random.permutation(train):
            ctc, tgt, valid = load_case(cnn, n, a.cut)
            # one slice at a time: 6.3 GB at batch 1 already
            x = torch.from_numpy(ctc)[None, None].to(dev)
            y = torch.from_numpy(tgt).amax(1)[None].to(dev)      # (1, 2, h, w), z collapsed
            v = torch.from_numpy(valid).amax(0)[None, None].to(dev)
            out = model(x)
            loss = (bce(out, y) * v).sum() / v.sum().clamp_min(1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss)
        print(f"epoch {ep}/{a.epochs} loss {tot/len(train):.4f} {time.time()-t0:.0f}s", flush=True)
        torch.save(model.state_dict(), work / "mjnet4d.pt")

    print("MJNET_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
