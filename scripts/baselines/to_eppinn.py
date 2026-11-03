"""Convert our ISLES npz cases into the per-case directory layout EPPINN reads.

EPPINN's loader (data.py:load_ctp_data) wants, per case:
    CTP_4D.nii.gz    4D volume, SimpleITK order (t, z, h, w), spacing[-1] = dt
    aif.npy          arterial input curve, length t
    brainmask.nii.gz 3D mask (z, h, w)
    vof.npy          OPTIONAL -- we have no venous output curve, so it is omitted and
                     EPPINN skips its partial-volume correction
    time.npy         OPTIONAL -- omitted, EPPINN derives time from the 4D spacing

Our data is split across two directories: cnn18/cnn24 hold `ctc` (z, t, h, w) and the
brain mask `valid`, while at2d18rx/at2d24rx hold the arterial curve `aif`. Both are keyed
by the same case name, so the converter reads one file from each.
Unlike SPPINN this needs no VOF and no 256x256 resampling -- EPPINN takes the geometry
from the image itself.

Cuts are applied by truncating the time axis, the same way the rest of the study does.
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DT = 1.0  # our frames are resampled to 1 s; EPPINN reads dt from spacing[-1]


def write_case(npz_path: Path, aif_path: Path, out_dir: Path, cut: int | None = None) -> None:
    z = np.load(npz_path)
    ctc = np.asarray(z["ctc"], dtype=np.float32)          # (z, t, h, w)
    mask = np.asarray(z["valid"], dtype=np.float32)       # (z, h, w)
    aif = np.asarray(np.load(aif_path)["aif"], dtype=np.float32)   # (t,)

    vol = np.transpose(ctc, (1, 0, 2, 3))                 # -> (t, z, h, w)
    if cut is not None:
        vol = vol[:cut]
        aif = aif[:cut]

    out_dir.mkdir(parents=True, exist_ok=True)

    # GetImageFromArray treats a 4D array as 3D-plus-vector unless told otherwise, which
    # leaves a 3D image with a 9-element direction. PyPeT needs a true 4D image.
    img = sitk.GetImageFromArray(vol, isVector=False)     # (t, z, h, w) -> 4D
    img.SetSpacing((1.0, 1.0, 1.0, DT))                   # last entry is the time step
    sitk.WriteImage(img, str(out_dir / "CTP_4D.nii.gz"))

    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    m.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(m, str(out_dir / "brainmask.nii.gz"))

    np.save(out_dir / "aif.npy", aif)


def main() -> None:
    src, aif_src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cut = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "full" else None
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    cases = [f for f in sorted(src.glob("*.npz")) if (aif_src / f.name).exists()]
    if n:
        cases = cases[:n]
    for f in cases:
        write_case(f, aif_src / f.name, dst / f.stem, cut)
    print(f"wrote {len(cases)} cases to {dst}" + (f" (cut {cut} s)" if cut else ""))


if __name__ == "__main__":
    main()
