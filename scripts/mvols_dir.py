import json, glob, sys, numpy as np
from pathlib import Path
d, out = sys.argv[1], sys.argv[2]; LS = ('25', '30', '35', 'full'); res = {}
for f in glob.glob(d + '/*.npz'):
    z = np.load(f); n = int(Path(f).stem.split('_')[1]); v = z['valid'].astype(float); pm = float(z['pixel_ml'])
    res[n] = {L: [float((z[f'mask_{L}'][:, j].astype(float) * v).sum() * pm) for j in (0, 1)] for L in LS}
json.dump(res, open(out, 'w')); print(out, len(res))
