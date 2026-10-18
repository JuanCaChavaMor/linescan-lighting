"""Geometric calibration of the line-scan camera from the checkerboard images in data/images (Section 4).

    python scripts/calibrate_linescan.py
"""
import json

import numpy as np
import pandas as pd

from _path import ROOT
from lslight.linescan import calibrate, joint_fit

SQUARE = 26.54                                   # mm, size of the checkerboard squares
paths = sorted((ROOT / "data" / "images").glob("*.png"))
rows = calibrate(paths, SQUARE)
per = pd.DataFrame(rows).drop(columns=["edges", "polarity"])

# repeatability: images 00000032 and 00000033 show the target at the same position
e = {r["image"]: r["edges"] for r in rows}
rep = e["00000032"] - e["00000033"]

# joint fit with target deviations, with and without the differently exposed image 00000037
fits = {}
for name, keep in (("all images", [r["image"] for r in rows]),
                   ("without 00000037", [r["image"] for r in rows if r["image"] != "00000037"])):
    fits[name] = joint_fit([e[k] for k in keep], SQUARE)
    per.loc[per.image.isin(keep), f"rms_joint ({name})"] = fits[name]["rms_per_image"]

per.to_csv(ROOT / "results" / "calibration_per_image.csv", index=False, float_format="%.4f")
summary = dict(
    square_mm=SQUARE,
    scale_px_per_mm=float(per.scale_mean.mean()), pixel_size_mm=float(1 / per.scale_mean.mean()),
    field_of_view_mm=float(4096 / per.scale_mean.mean()),
    repeatability_rms_px=float(np.sqrt(np.mean(rep ** 2))), repeatability_max_px=float(np.abs(rep).max()),
    rms_linear_px=float(per.rms_linear.mean()), rms_projective_px=float(per.rms_projective.mean()),
    joint={k: dict(rms_px=v["rms"], target_deviation_mm=v["deviation_mm"].round(4).tolist()) for k, v in fits.items()},
    edge_range_px=[float(per.first_edge.min()), float(per.last_edge.max())],
    saturated_share=dict(zip(per.image, per.saturated.round(3))),
)
(ROOT / "results" / "calibration.json").write_text(json.dumps(summary, indent=2))
print(per.round(3).to_string(index=False))
print(json.dumps(summary, indent=2))
