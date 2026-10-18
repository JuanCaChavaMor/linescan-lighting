"""Geometric calibration of a line-scan camera from images of a row of checkerboard squares.

A line-scan image is one sensor line repeated over time: every row of the image sees the same line of the target.
Averaging the rows gives a 1-D profile; the edges between dark and light squares are located with sub-pixel
precision (threshold crossing at mid-level, linear interpolation). With the square size s known, edge k of an image
lies at X_k = X_0 + k s on the target, and a line camera maps target positions to pixels by a 1-D projective map
(pinhole camera, target plane possibly tilted with respect to the sensor):
    u = (a + b X) / (1 + c X).
Per image we fit (a, b, c) by least squares; the residuals measure model error and edge noise, and the local scale
du/dX gives the pixel size on the target.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def read_profile(path):
    """Mean grey-level profile (rows averaged) of a line-scan image, and the share of saturated pixels."""
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(float)
    return g.mean(0), float(np.mean(g >= 255)), g


def edges(profile, frac=0.5):
    """Sub-pixel positions where the profile crosses the level between its dark and bright plateaus, with their
    polarity (+1 dark -> bright, -1 bright -> dark)."""
    lo, hi = np.percentile(profile, 2), np.percentile(profile, 98)
    t = lo + frac * (hi - lo)
    s = np.sign(profile - t)
    i = np.where(np.diff(s) != 0)[0]
    pos = i + (t - profile[i]) / (profile[i + 1] - profile[i])
    return pos, np.sign(profile[i + 1] - profile[i])


def projective(p, X):
    a, b, c = p
    return (a + b * X) / (1 + c * X)


def fit_image(u, square):
    """Fit u_k = projective(X_k) with X_k = k * square (k = 0..n-1). Returns params, residuals [px], scale [px/mm]."""
    X = np.arange(len(u)) * square
    p0 = [u[0], (u[-1] - u[0]) / X[-1], 0.0]
    r = least_squares(lambda p: projective(p, X) - u, p0)
    a, b, c = r.x
    dudX = (b - a * c) / (1 + c * X) ** 2
    return dict(params=r.x, resid=r.fun, scale=dudX, X=X)


def fit_linear(u, square):
    X = np.arange(len(u)) * square
    A = np.c_[np.ones_like(X), X]
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    return dict(params=coef, resid=A @ coef - u)


def calibrate(paths, square=26.54):
    rows = []
    for p in paths:
        prof, sat, _ = read_profile(p)
        u, pol = edges(prof)
        pr = fit_image(u, square)
        li = fit_linear(u, square)
        rows.append(dict(image=Path(p).stem.split("_")[0], n_edges=len(u), saturated=sat,
                         first_edge=u[0], last_edge=u[-1],
                         rms_linear=float(np.sqrt(np.mean(li["resid"] ** 2))),
                         rms_projective=float(np.sqrt(np.mean(pr["resid"] ** 2))),
                         scale_mean=float(np.mean(pr["scale"])), scale_change=float(pr["scale"][-1] / pr["scale"][0] - 1),
                         dark_minus_light_width=float(np.mean(np.diff(u)[pol[:-1] < 0]) -
                                                      np.mean(np.diff(u)[pol[:-1] > 0])),
                         edges=u, polarity=pol))
    return rows


def joint_fit(edge_lists, square=26.54):
    """All images together: each image has its own projective map (a_i, b_i, c_i), and the target has small
    deviations d_k of its edges from the nominal grid k * square, the same in every image (printing errors).

    The deviations are restricted to the space orthogonal to 1, k, k^2: a smooth deformation of the target is
    indistinguishable from a tilt or a scale of the projective map. Returns deviations [mm], per-image params,
    residuals [px] per image."""
    n = len(edge_lists[0])
    k = np.arange(n)
    X = k * square
    Q, _ = np.linalg.qr(np.c_[np.ones(n), k, k ** 2], mode="complete")
    B = Q[:, 3:]
    nb = B.shape[1]

    def res(p):
        d = B @ p[:nb]
        return np.concatenate([projective(p[nb + 3 * i:nb + 3 * i + 3], X + d) - u for i, u in enumerate(edge_lists)])

    p0 = np.r_[np.zeros(nb), np.concatenate([[u[0], (u[-1] - u[0]) / X[-1], 0.0] for u in edge_lists])]
    r = least_squares(res, p0)
    resid = r.fun.reshape(len(edge_lists), n)
    params = r.x[nb:].reshape(-1, 3)
    return dict(deviation_mm=B @ r.x[:nb], params=params, resid=resid,
                rms=float(np.sqrt(np.mean(resid ** 2))), rms_per_image=np.sqrt(np.mean(resid ** 2, axis=1)))
