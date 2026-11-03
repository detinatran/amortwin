"""Infer the trained 4D mJ-Net on the held-out fold and write per-patient volumes.

Output matches mvols_dir.py's format ({patient: {cut: [core_ml, hypo_ml]}}) so the study's
own scorers read it without special-casing. Volumes are probability-weighted sums over the
brain mask, never thresholded, as everywhere else in this work.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mjnet4d import MJNet4D

LS = ("25", "30", "35", "full")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cnn_dir"); p.add_argument("folds"); p.add_argument("work")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cut", type=int, default=30)
    a = p.parse_args()

    dev = "cuda"
    cnn = Path(a.cnn_dir); work = Path(a.work)
    folds = {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    def pid(n): return int(n.split("_")[1].split(".")[0])
    test = [f.name for f in sorted(cnn.glob("*.npz")) if folds.get(pid(f.name), -1) == a.fold]

    z0 = np.load(cnn / test[0])
    model = MJNet4D(z=z0["ctc"].shape[0], t=a.cut, out_ch=2).to(dev)
    model.load_state_dict(torch.load(work / "mjnet4d.pt", map_location=dev))
    model.eval()

    out = {}
    with torch.no_grad():
        for n in test:
            z = np.load(cnn / n)
            valid = np.asarray(z["valid"], dtype=np.float32)
            pm = float(z["pixel_ml"])
            per = {}
            for L in LS:
                k = None if L == "full" else int(L)
                ctc = np.asarray(z["ctc"], dtype=np.float32)
                ctc = ctc if k is None else ctc[:, :k]
                # the net was trained at one cut; shorter or longer input still runs because
                # Conv4D pads 'same' and the time axis is averaged away before the decoder
                x = torch.from_numpy(ctc)[None, None].to(dev)
                pr = torch.sigmoid(model(x))[0].cpu().numpy()      # (2, h, w)
                v2 = valid.max(0)                                   # (h, w)
                per[L] = [float((pr[j] * v2).sum() * pm) for j in (0, 1)]
            out[pid(n)] = per
            print(f"{n} ok", flush=True)

    (work / "mvols24.json").write_text(json.dumps(out))
    print(f"wrote {len(out)} patients", flush=True)
    print("MJNET_INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
