"""Irradiance along the scan line of a line-scan camera, lit by two sources placed symmetrically at its ends.

Coordinates (mm): the scan line is the x-axis on the inspection plane z = 0, the camera looks down from
(0, 0, z_cam), the light sources sit at x = +-d.

Two models:
* point sources (the first model of the project): E(x) = P z_s / ((x -+ d)^2 + z_s^2)^(3/2), the inverse-square
  law with the cosine of incidence;
* rectangular panels (the lab set-up): each panel, H x L mm, centred at (+-W/2, 0, z0), tilted by theta,
  emits like a generalised Lambertian source (radiance ~ cos^m of the emission angle, m = 1: Lambertian). The
  irradiance is the integral over the panel of  L cos^m(t_s) cos(t_r) / r^2 dA, by Gauss-Legendre quadrature.
"""
from dataclasses import dataclass, replace

import numpy as np


# ---------------------------------------------------------------- point sources
def point_sources(x, d, z_s, P=1.0):
    """Irradiance of two isotropic point sources of power P at (+-d, z_s) on the line z = 0."""
    return P * z_s * (((x - d) ** 2 + z_s ** 2) ** -1.5 + ((x + d) ** 2 + z_s ** 2) ** -1.5)


# ---------------------------------------------------------------- panels
@dataclass(frozen=True)
class PanelPair:
    W: float = 480.0          # distance between the panel centres [mm]
    z0: float = 209.1         # height of the panel centres above the inspection plane [mm]
    theta: float = 15.0       # tilt [deg]: 0 = facing straight down, > 0 = turned towards the centre of the line
    H: float = 210.0          # panel extent in the x-z plane [mm]
    L: float = 510.0          # panel extent along y (across the scan line) [mm]
    m: float = 1.0            # emission exponent (1 = Lambertian; larger = more directed, e.g. LED with lens)

    def with_(self, **kw):
        return replace(self, **kw)

    def geometry(self, side):
        """Centre, in-plane unit vector u (x-z plane) and normal n (into the scene) of the panel at side +1/-1."""
        t = np.radians(self.theta)
        c = np.array([side * self.W / 2, 0.0, self.z0])
        n = np.array([-side * np.sin(t), 0.0, -np.cos(t)])    # down, turned towards the centre for theta > 0
        u = np.array([side * np.cos(t), 0.0, -np.sin(t)])     # in-plane, perpendicular to n and to y
        return c, u, n

    def corners_x(self, side):
        c, u, _ = self.geometry(side)
        return [c + s * u * self.H / 2 for s in (-1, 1)]


def irradiance(x, panels=PanelPair(), nodes=(16, 24)):
    """Irradiance on the scan line points x (mm) from both panels, for unit radiance.

    Gauss-Legendre with nodes[0] points across the panel (u) and nodes[1] along it (y); only the part of each panel
    that faces the point contributes (cos of emission and incidence clipped at 0)."""
    x = np.atleast_1d(np.asarray(x, float))
    su, wu = np.polynomial.legendre.leggauss(nodes[0])
    sv, wv = np.polynomial.legendre.leggauss(nodes[1])
    su, wu = su * panels.H / 2, wu * panels.H / 2
    sv, wv = sv * panels.L / 2, wv * panels.L / 2
    E = np.zeros_like(x)
    p = np.stack([x, np.zeros_like(x), np.zeros_like(x)], -1)            # (nx, 3)
    for side in (+1, -1):
        c, u, n = panels.geometry(side)
        q = c + su[:, None, None] * u + sv[None, :, None] * np.array([0.0, 1.0, 0.0])   # (nu, nv, 3)
        d = p[:, None, None, :] - q[None]                                  # source -> receiver, (nx, nu, nv, 3)
        r = np.linalg.norm(d, axis=-1)
        cos_s = np.clip((d @ n) / r, 0, None)                              # emission angle at the panel
        cos_r = np.clip(-d[..., 2] / r, 0, None)                           # incidence angle at the plane
        E += np.einsum("xuv,u,v->x", cos_s ** panels.m * cos_r / r ** 2, wu, wv)
    return E


def emitted_flux(panels=PanelPair()):
    """Flux of both panels for unit radiance: 2 * area * 2 pi / (m + 1) (pi * area for a Lambertian panel)."""
    return 2 * panels.H * panels.L * 2 * np.pi / (panels.m + 1)


# ---------------------------------------------------------------- uniformity
def uniformity(E):
    """Standard measures of a profile E (> 0):
    U0 = min/mean (overall uniformity as in lighting standards), CV = std/mean, max/min."""
    E = np.asarray(E, float)
    return dict(U0=E.min() / E.mean(), CV=E.std() / E.mean(), ratio=E.max() / E.min(), mean=E.mean())
