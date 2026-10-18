"""Figures of the report and the README.

    python scripts/make_figures.py      # after the three study scripts
"""
import json

import numpy as np
import pandas as pd

from _path import ROOT
from lslight.illumination import PanelPair, irradiance, point_sources
from lslight.linescan import edges, read_profile
from lslight.plotting import C, INK, MUTED, plt, save

FIG, RES = ROOT / "figures", ROOT / "results"
ref = json.loads((RES / "design_reference.json").read_text())
WIDTH, Z_CAM = ref["width_mm"], ref["z_cam_mm"]
pareto = pd.read_csv(RES / "design_pareto.csv")

# ---------------------------------------------------------------- 1. geometry and profiles: lab vs optimised
lab = PanelPair()
b = pareto[(pareto.m == 1) & (pareto.min_eff_rel == 1.0)].iloc[0]
opt = lab.with_(W=b.W, z0=b.z0, theta=b.theta)
x = np.linspace(-WIDTH / 2, WIDTH / 2, 301)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), width_ratios=[1.1, 1])
for p, col, lab_ in ((lab, MUTED, "lab set-up"), (opt, C[0], "optimised, same efficiency")):
    for side in (1, -1):
        A, B = p.corners_x(side)
        a1.plot([A[0], B[0]], [A[2], B[2]], color=col, lw=4, solid_capstyle="butt",
                label=lab_ if side == 1 else None)
a1.plot([0, -WIDTH / 2], [Z_CAM, 0], color=INK, lw=0.8, ls="--")
a1.plot([0, WIDTH / 2], [Z_CAM, 0], color=INK, lw=0.8, ls="--", label="camera view")
a1.plot([-WIDTH / 2, WIDTH / 2], [0, 0], color=C[1], lw=3, label="scan line")
a1.plot(0, Z_CAM, "v", color=INK, ms=8)
a1.set_aspect("equal"); a1.set_xlabel("x [mm]"); a1.set_ylabel("z [mm]")
a1.legend(fontsize=7, loc="upper right")
for p, col, lab_ in ((lab, MUTED, "lab set-up"), (opt, C[0], "optimised")):
    E = irradiance(x, p)
    a2.plot(x, E / E.mean(), color=col, label=f"{lab_}: U0 = {E.min() / E.mean():.2f}")
a2.axhline(1, color=INK, lw=0.6)
a2.set_xlabel("position on the scan line x [mm]"); a2.set_ylabel("irradiance / mean")
a2.legend(fontsize=8)
save(fig, FIG / "design.png")

# ---------------------------------------------------------------- 2. Pareto front
lab_tilt = pd.read_csv(RES / "design_lab.csv")
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for k, m in enumerate((1.0, 3.0)):
    g = pareto[(pareto.m == m) & pareto.feasible]
    ax.plot(g.eff_rel, g.U0, "o-", color=C[k], label=f"optimised, emitter m = {m:g}")
t = lab_tilt[lab_tilt.case == "lab, tilt sweep"]
ax.plot(t.eff_rel, t.U0, ".", color=MUTED, label="lab geometry, tilt -15...60 deg")
ax.plot(t[t.blocks].eff_rel, t[t.blocks].U0, "x", color=C[1], label="blocks the camera's view")
l0 = lab_tilt[lab_tilt.case == "lab set-up"].iloc[0]
ax.plot(l0.eff_rel, l0.U0, "s", color=INK, label="lab set-up (15 deg)")
ax.set_xlabel("efficiency relative to the lab set-up"); ax.set_ylabel("uniformity U0 = min / mean")
ax.legend(fontsize=7, loc="lower left")
save(fig, FIG / "pareto.png")

# ---------------------------------------------------------------- 3. the images: saturation
paths = sorted((ROOT / "data" / "images").glob("*.png"))
prof, sat, g = read_profile(paths[0])
u, _ = edges(prof)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 3.6), height_ratios=[1, 1.4], sharex=True)
a1.imshow(g, cmap="gray", aspect="auto", vmin=0, vmax=255); a1.set_yticks([]); a1.grid(False)
a1.set_title(f"line-scan image {paths[0].stem.split('_')[0]} (50 lines): {100 * sat:.0f}% of the pixels at 255",
             loc="left", fontsize=8, color=INK)
a2.plot(prof, color=INK, lw=0.8)
a2.axhline(255, color=C[1], lw=1, ls="--", label="saturation (255)")
for e in u:
    a2.axvline(e, color=C[0], lw=0.6)
a2.set_xlabel("pixel"); a2.set_ylabel("grey level"); a2.legend(fontsize=7, loc="lower right")
save(fig, FIG / "images.png")

# ---------------------------------------------------------------- 4. calibration residuals and target deviations
cal = json.loads((RES / "calibration.json").read_text())
from lslight.linescan import fit_image, fit_linear, joint_fit  # noqa: E402
E = [edges(read_profile(p)[0])[0] for p in paths]
keep = [i for i, p in enumerate(paths) if not p.stem.startswith("00000037")]
jf = joint_fit([E[i] for i in keep], cal["square_mm"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), width_ratios=[1.4, 1])
k = np.arange(8)
for j, i in enumerate(keep):
    a1.plot(k, fit_linear(E[i], cal["square_mm"])["resid"], color=MUTED, lw=0.8, label="linear" if j == 0 else None)
    a1.plot(k, fit_image(E[i], cal["square_mm"])["resid"], color=C[0], lw=1, label="projective" if j == 0 else None)
    a1.plot(k, jf["resid"][j], color=C[1], lw=1, label="projective + target deviations" if j == 0 else None)
a1.axhline(0, color=INK, lw=0.6)
a1.set_xlabel("edge"); a1.set_ylabel("residual [px]"); a1.legend(fontsize=7)
a2.bar(k, 1000 * jf["deviation_mm"], color=C[2])
a2.set_xlabel("edge"); a2.set_ylabel("deviation of the target [µm]"); a2.grid(axis="x", visible=False)
save(fig, FIG / "calibration.png")

# ---------------------------------------------------------------- 5. point sources: absolute vs relative objective
ps = json.loads((RES / "point_sources.json").read_text())
xs = np.linspace(-5, 5, 400)
o, r = ps["original_objective"], ps["relative_objective"]
Io = 0.8 * (o["P1"] * o["h"] / ((xs - o["d"]) ** 2 + o["h"] ** 2) ** 1.5 +
            o["P2"] * o["h"] / ((xs + o["d"]) ** 2 + o["h"] ** 2) ** 1.5)
Ir = point_sources(xs, r["d"], r["h"])
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(xs, Io / Io.mean(), color=MUTED, label=f"min std: d = {o['d']:.0f}, h = {o['h']:.0f}, P at its lower bound "
                                                f"(U0 {o['U0']:.2f})")
ax.plot(xs, Ir / Ir.mean(), color=C[0], label=f"min std/mean: d = {r['d']:.2f}, h = {r['h']:.0f} (U0 {r['U0']:.2f})")
ax.set_xlabel("x [m]"); ax.set_ylabel("irradiance / mean"); ax.legend(fontsize=7)
save(fig, FIG / "point_sources.png")
print("figures written to", FIG)
