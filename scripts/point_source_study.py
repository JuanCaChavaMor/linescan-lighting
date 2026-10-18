"""The first model of the project: two point sources. Why the absolute standard deviation is the wrong objective.

    python scripts/point_source_study.py

The original scripts minimised sigma = std(I(x)) over the source powers, height and distance. sigma shrinks when
the light gets weaker, so the optimiser dims the sources and moves them away: the optimum sits on the bounds and
says nothing about uniformity. The relative measures (CV = sigma/mean, U0 = min/mean) do not depend on the power.
"""
import json

import numpy as np
from scipy.optimize import minimize

from _path import ROOT
from lslight.illumination import point_sources, uniformity

L, N, R = 10.0, 1000, 0.8                 # scan line [m], points, reflectivity (as in the original scripts)
x = np.linspace(-L / 2, L / 2, N)
BOUNDS = dict(P=(10, 200), h=(0.5, 5.0), d=(1.0, 10.0))


def profile(P1, P2, h, d):
    return R * (P1 * h / ((x - d) ** 2 + h ** 2) ** 1.5 + P2 * h / ((x + d) ** 2 + h ** 2) ** 1.5)


orig = minimize(lambda p: profile(*p).std(), [100, 100, 2, 5],
                bounds=[BOUNDS["P"], BOUNDS["P"], BOUNDS["h"], BOUNDS["d"]], method="L-BFGS-B")
starts = [(h0, d0) for h0 in (1, 2, 4) for d0 in (2, 5, 8)]
rel = min((minimize(lambda p: uniformity(point_sources(x, p[1], p[0]))["CV"], s,
                    bounds=[BOUNDS["h"], BOUNDS["d"]], method="L-BFGS-B") for s in starts), key=lambda r: r.fun)
# relative objective at fixed height (the height bound is the binding one): best distance for each height
heights = np.linspace(0.5, 5, 10)
by_height = []
for h in heights:
    r = min((minimize(lambda d: uniformity(point_sources(x, d[0], h))["CV"], [d0], bounds=[BOUNDS["d"]],
                      method="L-BFGS-B") for d0 in (1.5, 3, 5, 8)), key=lambda r: r.fun)
    u = uniformity(point_sources(x, r.x[0], h))
    by_height.append(dict(h=h, d=r.x[0], CV=u["CV"], U0=u["U0"], mean=u["mean"]))

Io = profile(*orig.x)
out = dict(
    original_objective=dict(P1=orig.x[0], P2=orig.x[1], h=orig.x[2], d=orig.x[3], sigma=orig.fun,
                            **{k: v for k, v in uniformity(Io).items()}),
    relative_objective=dict(h=rel.x[0], d=rel.x[1], **uniformity(point_sources(x, rel.x[1], rel.x[0]))),
    best_distance_by_height=by_height)
(ROOT / "results" / "point_sources.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "best_distance_by_height"}, indent=2, default=float))
