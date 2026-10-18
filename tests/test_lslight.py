import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lslight.design import blocks_view  # noqa: E402
from lslight.illumination import PanelPair, irradiance, point_sources, uniformity  # noqa: E402
from lslight.linescan import edges, joint_fit, projective  # noqa: E402


def test_point_sources_symmetric_and_scale_free():
    x = np.linspace(-5, 5, 101)
    E = point_sources(x, 3.0, 2.0, P=7.0)
    assert np.allclose(E, E[::-1])
    assert uniformity(E)["CV"] == pytest.approx(uniformity(point_sources(x, 3.0, 2.0, P=1.0))["CV"])


def test_small_panel_tends_to_point_source():
    # a tiny Lambertian panel facing down acts like a point source with a cosine-law emission
    p = PanelPair(W=200.0, z0=300.0, theta=0.0, H=1.0, L=1.0, m=1.0)
    x = np.linspace(-100, 100, 5)
    E = irradiance(x, p, nodes=(2, 2))
    r2 = (x[:, None] - np.array([100.0, -100.0])) ** 2 + 300.0 ** 2
    expected = (300.0 ** 2 / r2 ** 2).sum(1)                     # cos_s cos_r / r^2 = z^2 / r^4
    assert np.allclose(E, expected, rtol=1e-6)


def test_panel_quadrature_converges_and_is_symmetric():
    x = np.array([-200.0, 0.0, 200.0])
    a, b = irradiance(x, nodes=(8, 12)), irradiance(x, nodes=(32, 48))
    assert np.allclose(a, b, rtol=1e-4)
    assert a[0] == pytest.approx(a[2])


def test_blocking():
    assert not blocks_view(PanelPair(W=900, theta=30), width=491, z_cam=606.5)
    assert blocks_view(PanelPair(W=100, theta=0), width=491, z_cam=606.5)


def test_subpixel_edges_on_synthetic_profile():
    true = np.array([100.3, 300.7, 500.25])
    xs = np.arange(700)
    prof = np.full(700, 200.0)
    for a, b in ((true[0], true[1]),):
        prof = np.where((xs > a) & (xs < b), 20.0, prof)
    prof = np.where(xs > true[2], 20.0, prof)
    # linear ramps of one pixel at each edge, crossing the mid level exactly at the true position
    prof = np.interp(xs, np.sort(np.r_[xs, true - 0.5, true + 0.5]),
                     np.interp(np.sort(np.r_[xs, true - 0.5, true + 0.5]), xs, prof))
    u, pol = edges(prof)
    assert len(u) == 3 and np.allclose(u, true, atol=0.6)
    assert list(pol) == [-1, 1, -1]


def test_joint_fit_recovers_target_deviations():
    rng = np.random.default_rng(0)
    k = np.arange(8)
    dev = np.array([0.08, -0.05, -0.1, -0.02, 0.0, 0.14, 0.01, -0.07])
    Q, _ = np.linalg.qr(np.c_[np.ones(8), k, k ** 2], mode="complete")
    dev = Q[:, 3:] @ (Q[:, 3:].T @ dev)                        # the identifiable part
    X = k * 26.54 + dev
    imgs = [projective((u0, 8.34, c), X) + rng.normal(0, 0.02, 8)
            for u0, c in ((1000, 1e-5), (1200, -2e-5), (1400, 3e-5), (1100, 0.0))]
    fit = joint_fit(imgs, 26.54)
    assert np.allclose(fit["deviation_mm"], dev, atol=0.01)
    assert fit["rms"] < 0.03
