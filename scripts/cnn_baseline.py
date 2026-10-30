"""End-to-end CNN baseline with uncertainty: vendor core and hypoperfusion masks from a truncated CT perfusion.

python scripts/cnn_baseline.py train <cnn24_dir> <folds.json> <work_dir> [--epochs 60] [--fold k]
python scripts/cnn_baseline.py infer <cnn24_dir> <folds.json> <work_dir> [--mc 20]
python scripts/cnn_baseline.py eval  <work_dir> <refs24.json> <out.json> [--level 0.8]
python scripts/cnn_baseline.py infer <cnn18_dir> - <work_dir> --external cnn_volumes18.json
python scripts/cnn_baseline.py eval  <work_dir> <refs18.json> <out18.json> --volumes cnn_volumes18.json --calib cnn_volumes.json --calib-refs <refs24.json>

Inputs from cnn_export.py. A 2D U-Net (GroupNorm, dropout 0.2 in the bottleneck and decoder) sees one slice:
the F curve frames cut at a length L (frames beyond L set to zero) and the frame mask, 2F channels; it returns
the probability that a pixel's valid tissue is vendor core (rCBF < 0.30) and hypoperfused (Tmax > 6 s), trained
with BCE weighted by the valid share of the pixel against the full-scan labels. As AmorTwin, each training slice
is cut at L ~ U[25 s, its length] (the whole scan with probability 0.2). One model per patient fold.
Volumes are sum(p * valid) * pixel_ml. infer: the held-out patients of each fold at 25, 30, 35 s and full, one
deterministic pass (point) and --mc MC-dropout passes (draws). --external: every patient of another cohort through the five fold models (an ensemble: the point is the mean
of the five deterministic passes, the draws are the MC passes of all five). With --calib, eval applies the
conformal quantile of a calibration cohort (all its patients) instead of cross-fitting within the cohort.
eval: on ISLES'24, against the patient's own full
scan and the vendor volumes, the hypoperfusion interval of the draws with the normalised conformal calibration
and the selective DEFUSE 3 decision of prob_hybrid.py (cross-fitted by the same patient folds), and the same
with a constant conformal band on the point estimate; core and hypoperfusion errors, flips and agreement.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LENGTHS = ("25", "30", "35", "full")


def load_all(d):
    data = {}
    for f in sorted(Path(d).glob("training_*.npz")):
        z = np.load(f)
        data[int(f.stem.split("_")[1])] = {k: z[k] for k in ("ctc", "valid", "core", "hypo")} | {
            "frames": int(z["frames"]), "pixel_ml": float(z["pixel_ml"])}
    return data


def net(F, cin=None, cout=2):
    import torch.nn as nn

    def block(i, o, drop=0.0):
        layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                  nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU()]
        if drop:
            layers.append(nn.Dropout2d(drop))
        return nn.Sequential(*layers)

    class UNet(nn.Module):
        def __init__(self, c=(48, 96, 192, 384), drop=0.2):
            super().__init__()
            self.enc = nn.ModuleList([block(cin or 2 * F, c[0]), block(c[0], c[1]), block(c[1], c[2])])
            self.mid = block(c[2], c[3], drop)
            self.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[2], 2, 2), nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                     nn.ConvTranspose2d(c[1], c[0], 2, 2)])
            self.dec = nn.ModuleList([block(2 * c[2], c[2], drop), block(2 * c[1], c[1], drop), block(2 * c[0], c[0], drop)])
            self.head = nn.Conv2d(c[0], cout, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            import torch
            skips = []
            for e in self.enc:
                x = e(x)
                skips.append(x)
                x = self.pool(x)
            x = self.mid(x)
            for u, d, s in zip(self.up, self.dec, reversed(skips)):
                x = d(torch.cat([u(x), s], 1))
            return self.head(x)

    return UNet()


def cut(ctc, L):
    """ctc (..., F, S, S) float32 -> inputs (..., 2F, S, S) with the frames beyond L zeroed."""
    F = ctc.shape[-3]
    m = (np.arange(F) < L).astype(np.float32)[:, None, None]
    m = np.broadcast_to(m, ctc.shape)
    return np.concatenate([ctc * m, m], -3)


def train(a):
    import torch
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load_all(a.data)
    folds = {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    F = next(iter(data.values()))["ctc"].shape[1]
    dev = "cuda"
    a.work.mkdir(parents=True, exist_ok=True)
    for k in ([a.fold] if a.fold is not None else range(5)):
        wf = a.work / f"cnn_fold{k}.pt"
        if wf.exists():
            continue
        slices = [(n, z) for n, d in data.items() if folds.get(n, -1) != k for z in range(d["ctc"].shape[0])
                  if d["valid"][z].sum() > 50]
        model = net(F).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = a.epochs * (len(slices) // a.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        t0, step = time.time(), 0
        for ep in range(a.epochs):
            order = rng.permutation(len(slices))
            for b in range(len(slices) // a.bs):
                X, Y, W = [], [], []
                for i in order[b * a.bs:(b + 1) * a.bs]:
                    n, z = slices[i]
                    d = data[n]
                    L = d["frames"] if rng.random() < 0.2 else int(rng.integers(25, d["frames"] + 1))
                    x = cut(d["ctc"][z].astype(np.float32), L)
                    v = d["valid"][z].astype(np.float32)
                    y = np.stack([d["core"][z], d["hypo"][z]]).astype(np.float32) / np.maximum(v, 1e-3)
                    if rng.random() < 0.5:
                        x, y, v = x[..., ::-1], y[..., ::-1], v[..., ::-1]
                    X.append(x), Y.append(np.clip(y, 0, 1)), W.append(v)
                X, Y, W = (torch.as_tensor(np.ascontiguousarray(np.stack(t))).to(dev) for t in (X, Y, W))
                logit = model(X)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, Y, reduction="none")
                        * W[:, None]).sum() / W.sum().clamp_min(1) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                step += 1
            print(f"fold {k} epoch {ep + 1}/{a.epochs} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), wf)


def infer(a):
    import torch
    data = load_all(a.data)
    if a.shift:
        for d in data.values():
            F = d["ctc"].shape[1]
            d["ctc"] = np.concatenate([d["ctc"][:, a.shift:], np.zeros_like(d["ctc"][:, :a.shift])], 1)
            d["frames"] -= a.shift
    folds = {} if a.external else {int(k): v for k, v in json.loads(Path(a.folds).read_text()).items()}
    F = next(iter(data.values()))["ctc"].shape[1]
    dev = "cuda"
    out = {}
    if a.external:
        folds = {n: -1 for n in data}
    for k in range(5):
        model = net(F).to(dev)
        model.load_state_dict(torch.load(a.work / f"cnn_fold{k}.pt", map_location=dev))
        for n, d in data.items():
            if folds.get(n, -1) != k and not a.external:
                continue
            v = torch.as_tensor(d["valid"].astype(np.float32)).to(dev)
            rec = {}
            for Ls in LENGTHS:
                L = d["frames"] if Ls == "full" else int(Ls)
                X = torch.as_tensor(cut(d["ctc"].astype(np.float32), L)).to(dev)
                vols = []
                with torch.no_grad():
                    for it in range(a.mc + 1):
                        model.train(it > 0)             # pass 0: dropout off (point); then MC dropout
                        p = torch.sigmoid(model(X))
                        vols.append([float((p[:, j] * v).sum()) * d["pixel_ml"] for j in (0, 1)])
                rec[Ls] = vols
            if a.external:
                out.setdefault(n, {L: [] for L in LENGTHS})
                for L in LENGTHS:
                    out[n][L].append(rec[L])
            else:
                out[n] = rec
                print(n, {L: [round(x, 1) for x in rec[L][0]] for L in LENGTHS}, flush=True)
    if a.external:
        for n in out:         # ensemble: point = mean of the five points, draws = all MC passes
            out[n] = {L: [list(np.mean([m[0] for m in out[n][L]], 0))] + [x for m in out[n][L] for x in m[1:]]
                      for L in LENGTHS}
    (a.work / (a.external or "cnn_volumes.json")).write_text(json.dumps(out))


def evaluate(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from truncation_accuracy import defuse3, folds as make_folds
    def cohort(vfile, rfile):
        vol = {int(n): r for n, r in json.loads((a.work / vfile).read_text()).items()}
        refs = json.loads(Path(rfile).read_text())
        cr, hr, dr = ({int(n): v for n, v in refs[k].items()} for k in ("core", "hypo", "dec"))
        groups = {int(n): g for n, g in refs["groups"].items()}
        return vol, cr, hr, dr, groups

    vol, core_ref, hypo_ref, dec_ref, groups = cohort(a.volumes, a.refs)
    pats = sorted(n for n in vol if n in core_ref)
    fold = make_folds({n: groups[n] for n in pats})
    cal = cohort(a.calib, a.calib_refs) if a.calib else None
    d3 = lambda c, h: defuse3(c, max(c, h))  # noqa: E731
    qlo, qhi = (1 - a.level) / 2, (1 + a.level) / 2
    out = {"n": len(pats)}
    for Ls in LENGTHS:
        res = {}
        pt = {n: vol[n][Ls][0] for n in pats}
        full = {n: vol[n]["full"][0] for n in pats}
        res["core_mae_ml"] = float(np.median([abs(pt[n][0] - core_ref[n]) for n in pats]))
        res["hypo_mae_ml"] = float(np.median([abs(pt[n][1] - hypo_ref[n]) for n in pats]))
        flip = np.array([d3(*pt[n]) != d3(*full[n]) for n in pats])
        agree = np.array([d3(*pt[n]) == dec_ref[n] for n in pats])
        res["flips"], res["agree"] = float(flip.mean()), float(agree.mean())
        if Ls != "full":
            for ref_kind in ("self", "vendor"):
                for mode in ("mc_norm", "const"):
                    def records(V, HR, ns):
                        rec = {}
                        for n in ns:
                            h = np.array([x[1] for x in V[n][Ls][1:]])
                            y = np.log1p(V[n]["full"][0][1] if ref_kind == "self" else HR[n])
                            if mode == "mc_norm":
                                lo, hi = np.log1p(np.quantile(h, qlo)), np.log1p(np.quantile(h, qhi))
                                w = max(hi - lo, 0.05)
                            else:
                                lo = hi = np.log1p(V[n][Ls][0][1])
                                w = 1.0
                            rec[n] = (lo, hi, w, y, max(lo - y, y - hi) / w)
                        return rec

                    rec = records(vol, hypo_ref, pats)
                    if cal:
                        cpats = sorted(n for n in cal[0] if n in cal[1])
                        cal_sc = [r[4] for r in records(cal[0], cal[2], cpats).values()]
                    cov, wid, cert = [], [], []
                    for n in pats:
                        sc = cal_sc if cal else [rec[m][4] for m in pats if fold[m] != fold[n]]
                        q = float(np.quantile(sc, min(1.0, a.level * (1 + 1 / len(sc)))))
                        lo, hi, w, y, _ = rec[n]
                        lo, hi = lo - q * w, hi + q * w
                        cov.append(lo <= y <= hi)
                        wid.append(float(np.expm1(hi) - np.expm1(max(lo, 0.0))))
                        c = pt[n][0]
                        cert.append(d3(c, float(np.expm1(max(lo, 0.0)))) == d3(c, float(np.expm1(hi))))
                    cert = np.array(cert)
                    res[f"{ref_kind}_{mode}"] = {
                        "cover": float(np.mean(cov)), "width_median_ml": float(np.median(wid)),
                        "certain": float(cert.mean()),
                        "flip_certain": float(flip[cert].mean()) if cert.any() else None,
                        "flip_uncertain": float(flip[~cert].mean()) if (~cert).any() else None,
                        "agree_certain": float(agree[cert].mean()) if cert.any() else None}
        out[Ls] = res
        print(Ls, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                              if not isinstance(v, dict)}),
              *[f"\n   {k}: " + json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                for k, v in res.items() if isinstance(v, dict)])
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("train", "infer", "eval"))
    p.add_argument("args", nargs="+")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--fold", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mc", type=int, default=20)
    p.add_argument("--level", type=float, default=0.8)
    p.add_argument("--shift", type=int, default=0, help="infer: drop the first k frames (a later start)")
    p.add_argument("--external", help="infer every patient with all five fold models; output file name")
    p.add_argument("--volumes", default="cnn_volumes.json")
    p.add_argument("--calib", help="volumes of the calibration cohort (in work_dir)")
    p.add_argument("--calib-refs")
    a = p.parse_args()
    if a.stage == "eval":
        a.work, a.refs, a.out = Path(a.args[0]), a.args[1], a.args[2]
        evaluate(a)
        return
    a.data, a.folds, a.work = Path(a.args[0]), a.args[1], Path(a.args[2])
    (train if a.stage == "train" else infer)(a)


if __name__ == "__main__":
    main()
