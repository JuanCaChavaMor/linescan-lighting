"""Illumination design with the panel model (Section 3).

    python scripts/design_study.py      # ~2 min

1. the lab set-up (W = 480 mm, z0 = 209.1 mm, theta = 15 deg, panels 210 x 510 mm, camera at 606.5 mm), inspected
   width = the measured field of view (491 mm, scripts/calibrate_linescan.py);
2. effect of the tilt alone;
3. Pareto front uniformity vs efficiency: maximise U0 over (W, z0, theta) for a range of minimum efficiencies,
   with the camera's view kept free; for Lambertian (m = 1) and more directed (m = 3) emitters.
"""
import json

import numpy as np
import pandas as pd

from _path import ROOT
from lslight.design import evaluate, optimize
from lslight.illumination import PanelPair

WIDTH, Z_CAM = 491.0, 606.5
RES = ROOT / "results"
lab = PanelPair()
ref = evaluate(lab, WIDTH, Z_CAM)["eff"]


def row(p, **extra):
    e = evaluate(p, WIDTH, Z_CAM, ref)
    return dict(W=p.W, z0=p.z0, theta=p.theta, m=p.m, U0=e["U0"], CV=e["CV"], ratio=e["ratio"],
                eff_rel=e["eff_rel"], blocks=bool(e["blocks"]), **extra)


rows = [row(lab, case="lab set-up")]
rows += [row(lab.with_(theta=t), case="lab, tilt sweep") for t in np.arange(-15, 61, 5)]
pd.DataFrame(rows).to_csv(RES / "design_lab.csv", index=False, float_format="%.4f")
print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

front = []
for m in (1.0, 3.0):
    base = lab.with_(m=m)
    ref_m = evaluate(base, WIDTH, Z_CAM)["eff"]          # efficiency relative to the lab geometry with that emitter
    for e_min in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        best, ev, _ = optimize(base, WIDTH, Z_CAM, min_eff_rel=e_min, ref_eff=ref_m, maxiter=40)
        r = dict(m=m, min_eff_rel=e_min, W=best.W, z0=best.z0, theta=best.theta, U0=ev["U0"], CV=ev["CV"],
                 eff_rel=ev["eff_rel"], blocks=bool(ev["blocks"]),
                 feasible=bool(not ev["blocks"] and ev["eff_rel"] >= e_min - 1e-3))
        front.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(front).to_csv(RES / "design_pareto.csv", index=False, float_format="%.4f")
(RES / "design_reference.json").write_text(json.dumps(dict(width_mm=WIDTH, z_cam_mm=Z_CAM, lab=rows[0]), indent=2,
                                                      default=float))
