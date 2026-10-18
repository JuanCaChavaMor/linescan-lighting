"""Design of the two-panel illumination: choose the geometry (W, z0, theta) for a uniform and efficient scan line.

The emission of the panels (m) and their size are given by the hardware; what can be chosen is where they are.
Two goals conflict: uniformity (U0 = min/mean over the inspected width) and efficiency (how much of the emitted
flux reaches the scan line, here the mean irradiance per unit emitted flux, relative to the lab set-up). A design
must also leave the camera's view of the scan line free.
"""
import numpy as np
from scipy.optimize import differential_evolution

from .illumination import PanelPair, emitted_flux, irradiance, uniformity

BOUNDS = dict(W=(300.0, 1200.0), z0=(30.0, 450.0), theta=(-10.0, 75.0))


def blocks_view(panels, width, z_cam, n=50):
    """True if a panel (which spans y = 0) enters the cone in which the camera at (0, 0, z_cam) sees the inspected
    width [-width/2, width/2], or reaches below the inspection plane."""
    for side in (+1, -1):
        a, b = panels.corners_x(side)
        pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
        if np.any(pts[:, 2] <= 5.0):
            return True
        inside = np.abs(pts[:, 0]) < width / 2 * (1 - pts[:, 2] / z_cam)
        if np.any(inside & (pts[:, 2] < z_cam)):
            return True
    return False


def evaluate(panels, width, z_cam, ref_eff=None, n=121):
    x = np.linspace(-width / 2, width / 2, n)
    E = irradiance(x, panels, nodes=(10, 14))
    u = uniformity(E)
    u["eff"] = u["mean"] / emitted_flux(panels)
    if ref_eff:
        u["eff_rel"] = u["eff"] / ref_eff
    u["blocks"] = blocks_view(panels, width, z_cam)
    return u


def optimize(base, width, z_cam, min_eff_rel=0.0, ref_eff=None, objective="U0", seed=0, maxiter=60):
    """Maximise U0 (or minimise CV) over (W, z0, theta) subject to: no view blocking, efficiency >= min_eff_rel x
    reference. Constraints by penalty; global search by differential evolution, polished by L-BFGS-B."""
    keys = list(BOUNDS)

    def f(v):
        p = base.with_(**dict(zip(keys, v)))
        e = evaluate(p, width, z_cam, ref_eff)
        pen = 10.0 * e["blocks"] + 10.0 * max(0.0, min_eff_rel - e.get("eff_rel", 1.0))
        return (-e["U0"] if objective == "U0" else e["CV"]) + pen

    res = differential_evolution(f, [BOUNDS[k] for k in keys], seed=seed, maxiter=maxiter, tol=1e-7, popsize=15,
                                 polish=True)
    best = base.with_(**dict(zip(keys, res.x)))
    return best, evaluate(best, width, z_cam, ref_eff), res
