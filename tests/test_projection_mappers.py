# Copyright (c) 2026 Simone Coniglio
"""Tests for the projection mapper classes: Free2D, Free3D, ALM2D, ALM3D.

All mappers are pure-NumPy (no FEniCS dependency).
"""
import numpy as np
import pytest

from ggp.projection.registry import get_mapper, list_mappers
from ggp.projection.free_2d import Free2DMapper
from ggp.projection.free_3d import Free3DMapper
from ggp.projection.alm_2d import ALM2DMapper
from ggp.projection.alm_3d import ALM3DMapper


# ── helpers ───────────────────────────────────────────────────────────────────

def _grid_2d(n=4):
    """n² evaluation points in (0,1)²."""
    x = np.linspace(0.15, 0.85, n)
    xx, yy = np.meshgrid(x, x)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _grid_3d(n=3):
    """n³ evaluation points in (0,1)³."""
    x = np.linspace(0.15, 0.85, n)
    xx, yy, zz = np.meshgrid(x, x, x)
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _fd_check(mapper_forward, x0, eval_coords, comp, eps=1e-5):
    """Return finite-difference gradient of fE[comp] w.r.t. x0."""
    fE0, _ = mapper_forward(x0, eval_coords)
    fd = np.zeros((len(x0), eval_coords.shape[0]))
    for j in range(len(x0)):
        xp = x0.copy(); xp[j] += eps
        fEp, _ = mapper_forward(xp, eval_coords)
        fd[j] = (fEp[comp] - fE0[comp]) / eps
    return fd


# ── Free2DMapper ──────────────────────────────────────────────────────────────

class TestFree2DMapper:

    def test_num_vars_per_component(self):
        assert Free2DMapper(num_components=1).num_vars_per_component() == 6

    def test_default_bounds_shape_and_ordering(self):
        m = Free2DMapper(num_components=3)
        lb, ub = m.default_bounds((10.0, 5.0), 3)
        assert lb.shape == (18,)
        assert ub.shape == (18,)
        assert np.all(ub >= lb)

    def test_registry_aliases(self):
        assert isinstance(get_mapper("Free", num_components=1), Free2DMapper)
        assert isinstance(get_mapper("2D_Free", num_components=1), Free2DMapper)

    def test_forward_shape(self):
        m = Free2DMapper(num_components=2)
        ec = _grid_2d()
        x = np.tile([0.5, 0.5, 0.3, 0.2, 0.0, 1.0], 2)
        fE, fV = m.forward(x, ec)
        assert fE.shape == (2, len(ec))
        assert fV.shape == (2, len(ec))

    def test_forward_non_negative(self):
        m = Free2DMapper(num_components=2)
        ec = _grid_2d()
        x = np.tile([0.5, 0.5, 0.3, 0.2, 0.0, 1.0], 2)
        fE, fV = m.forward(x, ec)
        assert np.all(fE >= 0)
        assert np.all(fV >= 0)

    def test_forward_power_scaling(self):
        """Squaring the density variable (power_E=2) reduces contribution."""
        m = Free2DMapper(num_components=1)
        ec = _grid_2d(n=3)
        x = np.array([0.5, 0.5, 0.4, 0.2, 0.0, 0.7])
        fE1, _ = m.forward(x, ec, power_E=1.0)
        fE2, _ = m.forward(x, ec, power_E=2.0)
        assert np.all(fE2 <= fE1 + 1e-10)  # Mc=0.7 → Mc² < Mc

    def test_forward_zero_density(self):
        m = Free2DMapper(num_components=1)
        ec = _grid_2d(n=3)
        x = np.array([0.5, 0.5, 0.4, 0.2, 0.0, 0.0])  # Mc=0
        fE, fV = m.forward(x, ec)
        np.testing.assert_allclose(fE, 0.0, atol=1e-10)

    def test_jacobian_keys_and_shapes(self):
        m = Free2DMapper(num_components=2)
        ec = _grid_2d()
        x = np.tile([0.5, 0.5, 0.3, 0.2, 0.0, 1.0], 2)
        jac = m.jacobian(x, ec)
        assert {"funcs_E", "funcs_V", "grads_E", "grads_V"} <= jac.keys()
        assert jac["grads_E"].shape == (2, 6, len(ec))
        assert jac["grads_V"].shape == (2, 6, len(ec))

    def test_jacobian_finite_difference(self):
        """Analytic Jacobian matches central finite differences for all 6 vars."""
        m = Free2DMapper(num_components=1)
        ec = _grid_2d(n=3)
        x0 = np.array([0.55, 0.50, 0.35, 0.18, 0.25, 0.85])
        jac = m.jacobian(x0, ec)
        grads_E = jac["grads_E"][0]  # (6, n_el)
        eps = 1e-5
        for j in range(6):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            fEp, _ = m.forward(xp, ec)
            fEm, _ = m.forward(xm, ec)
            fd = (fEp[0] - fEm[0]) / (2 * eps)
            np.testing.assert_allclose(
                grads_E[j], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"Free2D grad mismatch at var {j}")

    def test_jacobian_finite_difference_volume(self):
        """Volume Jacobian (power_V=1) is consistent with forward."""
        m = Free2DMapper(num_components=1)
        ec = _grid_2d(n=3)
        x0 = np.array([0.4, 0.6, 0.3, 0.15, 0.1, 0.9])
        jac = m.jacobian(x0, ec)
        grads_V = jac["grads_V"][0]  # (6, n_el)
        eps = 1e-5
        for j in range(6):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            _, fVp = m.forward(xp, ec)
            _, fVm = m.forward(xm, ec)
            fd = (fVp[0] - fVm[0]) / (2 * eps)
            np.testing.assert_allclose(
                grads_V[j], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"Free2D vol grad mismatch at var {j}")

    def test_wgp_method(self):
        m = Free2DMapper(num_components=1, method='WGP')
        ec = _grid_2d(n=3)
        x = np.array([0.5, 0.5, 0.3, 0.2, 0.0, 1.0])
        fE, fV = m.forward(x, ec)
        assert fE.shape == (1, len(ec))

    def test_jacobian_power_e_zero(self):
        """power_E=0 special-case: gradient of density var should be 0."""
        m = Free2DMapper(num_components=1)
        ec = _grid_2d(n=3)
        x0 = np.array([0.5, 0.5, 0.3, 0.2, 0.0, 0.8])
        jac = m.jacobian(x0, ec, power_E=0.0)
        # d(W * Mc^0)/dMc = 0
        np.testing.assert_allclose(jac["grads_E"][0, 5], 0.0, atol=1e-10)


# ── Free3DMapper ──────────────────────────────────────────────────────────────

class TestFree3DMapper:

    def test_num_vars_per_component(self):
        assert Free3DMapper(num_components=1).num_vars_per_component() == 8

    def test_default_bounds_shape(self):
        m = Free3DMapper(num_components=2)
        lb, ub = m.default_bounds((10.0, 5.0, 4.0), 2)
        assert lb.shape == (16,)
        assert ub.shape == (16,)
        assert np.all(ub >= lb)

    def test_registry(self):
        assert isinstance(get_mapper("3D_Free", num_components=1), Free3DMapper)

    def test_forward_shape_and_range(self):
        m = Free3DMapper(num_components=2)
        ec = _grid_3d()
        x = np.tile([0.5, 0.5, 0.5, 0.3, 0.2, 0.0, 0.0, 1.0], 2)
        fE, fV = m.forward(x, ec)
        assert fE.shape == (2, len(ec))
        assert fV.shape == (2, len(ec))
        assert np.all(fE >= 0)

    def test_forward_zero_density(self):
        m = Free3DMapper(num_components=1)
        ec = _grid_3d(n=2)
        x = np.array([0.5, 0.5, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0])  # Mc=0
        fE, _ = m.forward(x, ec)
        np.testing.assert_allclose(fE, 0.0, atol=1e-10)

    def test_jacobian_shape(self):
        m = Free3DMapper(num_components=1)
        ec = _grid_3d()
        x = np.array([0.5, 0.5, 0.5, 0.3, 0.2, 0.0, 0.0, 1.0])
        jac = m.jacobian(x, ec)
        assert jac["grads_E"].shape == (1, 8, len(ec))
        assert jac["grads_V"].shape == (1, 8, len(ec))

    def test_jacobian_finite_difference(self):
        m = Free3DMapper(num_components=1)
        ec = _grid_3d(n=2)
        x0 = np.array([0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.15, 0.85])
        jac = m.jacobian(x0, ec)
        grads_E = jac["grads_E"][0]  # (8, n_el)
        eps = 1e-5
        for j in range(8):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            fEp, _ = m.forward(xp, ec)
            fEm, _ = m.forward(xm, ec)
            fd = (fEp[0] - fEm[0]) / (2 * eps)
            np.testing.assert_allclose(
                grads_E[j], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"Free3D grad mismatch at var {j}")

    def test_jacobian_volume_finite_difference(self):
        m = Free3DMapper(num_components=1)
        ec = _grid_3d(n=2)
        x0 = np.array([0.5, 0.5, 0.5, 0.25, 0.18, 0.0, 0.0, 0.9])
        jac = m.jacobian(x0, ec)
        grads_V = jac["grads_V"][0]
        eps = 1e-5
        for j in range(8):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            _, fVp = m.forward(xp, ec)
            _, fVm = m.forward(xm, ec)
            fd = (fVp[0] - fVm[0]) / (2 * eps)
            np.testing.assert_allclose(
                grads_V[j], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"Free3D vol grad mismatch at var {j}")


# ── ALM2DMapper ───────────────────────────────────────────────────────────────

class TestALM2DMapper:
    """Tests for ALM2DMapper with the MNA continuous variable layout:
      [Xc_0_0, L_0_0, ..., Xc_{nY-1}_{np-1}, L_{nY-1}_{np-1},
       h_0..h_{np-1},  Mc_0..Mc_{np-1},  y0, theta0]
    """

    NL = 3   # num_layers
    CPL = 2  # comp_per_layer (= number of printed components)
    LH = 1.0 / 3.0  # layer_height

    def _mapper(self):
        return ALM2DMapper(
            num_components=self.NL * self.CPL,
            num_layers=self.NL,
            comp_per_layer=self.CPL,
            layer_height=self.LH,
        )

    def _x0(self, m):
        """Feasible test design vector in unscaled physical space."""
        n = m.num_vars()                           # 2*NL*CPL + 2*CPL + 2
        nY, np_val = self.NL, self.CPL
        n_xl = 2 * nY * np_val
        x = np.zeros(n)
        x[0:n_xl:2] = 0.5 * np_val                # Xc: mid-domain
        x[1:n_xl:2] = 0.1                          # L: thin
        x[n_xl        : n_xl + np_val] = 0.8       # h
        x[n_xl + np_val : n_xl + 2*np_val] = 0.7  # Mc
        x[n_xl + 2*np_val] = 0.0                   # y0
        x[n_xl + 2*np_val + 1] = 0.0               # theta0
        return x

    def test_num_vars(self):
        # 2*3*2 + 2*2 + 2 = 18
        assert self._mapper().num_vars() == 2 * self.NL * self.CPL + 2 * self.CPL + 2

    def test_default_bounds_shape(self):
        m = self._mapper()
        lb, ub = m.default_bounds((10.0, 1.0), self.NL * self.CPL)
        assert lb.shape == (m.num_vars(),)
        assert ub.shape == (m.num_vars(),)

    def test_registry_aliases(self):
        kw = dict(num_components=6, num_layers=3, comp_per_layer=2, layer_height=1.0/3.0)
        assert isinstance(get_mapper("ALM", **kw), ALM2DMapper)
        assert isinstance(get_mapper("2D_ALM", **kw), ALM2DMapper)

    def test_forward_shape_and_range(self):
        m = self._mapper()
        ec = _grid_2d()
        x = self._x0(m)
        fE, fV = m.forward(x, ec)
        # Output is (CPL, n_elements) — one char-func per printed component
        assert fE.shape == (self.CPL, len(ec))
        assert fV.shape == (self.CPL, len(ec))
        assert np.all(fE >= 0)
        assert np.all(fV >= 0)

    def test_jacobian_shape_and_is_continuous_flag(self):
        m = self._mapper()
        ec = _grid_2d()
        x = self._x0(m)
        jac = m.jacobian(x, ec)
        assert jac["is_continuous"] is True
        # grads use global-variable column indexing: (CPL, n_total_vars, n_elements)
        n_tot = m.num_vars()
        assert jac["grads_E"].shape == (self.CPL, n_tot, len(ec))
        assert jac["grads_V"].shape == (self.CPL, n_tot, len(ec))

    def test_jacobian_finite_difference(self):
        """Analytic gradient matches central FD for all design variables."""
        m = ALM2DMapper(num_components=2, num_layers=2, comp_per_layer=1, layer_height=0.5)
        ec = _grid_2d(n=3)
        nY, np_val = 2, 1
        n_xl = 2 * nY * np_val   # 4
        n_tot = n_xl + 2*np_val + 2  # 8
        # x0: [Xc_0_0, L_0_0, Xc_1_0, L_1_0, h_0, Mc_0, y0, theta0]
        x0 = np.array([0.50, 0.15, 0.50, 0.15, 0.80, 0.70, 0.0, 0.0])
        jac = m.jacobian(x0, ec)
        grads_E = jac["grads_E"]   # (1, 8, n_el)
        eps = 1e-5
        for v in range(n_tot):
            xp = x0.copy(); xp[v] += eps
            xm = x0.copy(); xm[v] -= eps
            fEp, _ = m.forward(xp, ec)
            fEm, _ = m.forward(xm, ec)
            fd = (fEp[0] - fEm[0]) / (2 * eps)   # component 0 only (CPL=1)
            np.testing.assert_allclose(
                grads_E[0, v], fd, rtol=5e-3, atol=1e-5,
                err_msg=f"ALM2D MNA grad mismatch at var {v}")


# ── ALM3DMapper ───────────────────────────────────────────────────────────────

class TestALM3DMapper:

    NL = 2
    CPL = 2
    LH = 0.5

    def _mapper(self):
        return ALM3DMapper(
            num_components=self.NL * self.CPL,
            num_layers=self.NL,
            comp_per_layer=self.CPL,
            layer_height=self.LH,
        )

    def test_num_vars_per_component(self):
        assert self._mapper().num_vars_per_component() == 6

    def test_default_bounds_shape(self):
        m = self._mapper()
        lb, ub = m.default_bounds((10.0, 5.0, 4.0), self.NL * self.CPL)
        assert lb.shape == (self.NL * self.CPL * 6,)
        assert ub.shape == (self.NL * self.CPL * 6,)

    def test_registry(self):
        m = get_mapper("3D_ALM", num_components=4, num_layers=2,
                       comp_per_layer=2, layer_height=0.5)
        assert isinstance(m, ALM3DMapper)

    def test_forward_shape_and_range(self):
        m = self._mapper()
        ec = _grid_3d()
        x = np.tile([0.5, 0.5, 0.3, 0.3, 0.0, 0.9], self.NL * self.CPL)
        fE, fV = m.forward(x, ec)
        assert fE.shape == (self.NL * self.CPL, len(ec))
        assert fV.shape == (self.NL * self.CPL, len(ec))
        assert np.all(fE >= 0)

    def test_forward_zero_density(self):
        m = ALM3DMapper(num_components=1, num_layers=1, comp_per_layer=1, layer_height=0.5)
        ec = _grid_3d(n=2)
        x = np.array([0.5, 0.5, 0.3, 0.3, 0.0, 0.0])  # Mc=0
        fE, _ = m.forward(x, ec)
        np.testing.assert_allclose(fE, 0.0, atol=1e-10)

    def test_jacobian_shape(self):
        m = self._mapper()
        ec = _grid_3d()
        x = np.tile([0.5, 0.5, 0.3, 0.3, 0.0, 0.9], self.NL * self.CPL)
        jac = m.jacobian(x, ec)
        assert jac["grads_E"].shape == (self.NL * self.CPL, 6, len(ec))
        assert jac["grads_V"].shape == (self.NL * self.CPL, 6, len(ec))

    def test_jacobian_finite_difference(self):
        m = ALM3DMapper(num_components=2, num_layers=1, comp_per_layer=2, layer_height=0.5)
        ec = _grid_3d(n=2)
        x0 = np.tile([0.5, 0.5, 0.3, 0.25, 0.1, 0.85], 2)
        jac = m.jacobian(x0, ec)
        grads_E = jac["grads_E"]   # (2, 6, n_el)
        eps = 1e-5
        for comp in range(2):
            for v in range(6):
                j = comp * 6 + v
                xp = x0.copy(); xp[j] += eps
                xm = x0.copy(); xm[j] -= eps
                fEp, _ = m.forward(xp, ec)
                fEm, _ = m.forward(xm, ec)
                fd = (fEp[comp] - fEm[comp]) / (2 * eps)
                np.testing.assert_allclose(
                    grads_E[comp, v], fd, rtol=1e-3, atol=1e-5,
                    err_msg=f"ALM3D grad mismatch comp={comp}, var={v}")

    def test_jacobian_volume_finite_difference(self):
        m = ALM3DMapper(num_components=1, num_layers=1, comp_per_layer=1, layer_height=0.5)
        ec = _grid_3d(n=2)
        x0 = np.array([0.5, 0.5, 0.3, 0.25, 0.0, 0.8])
        jac = m.jacobian(x0, ec)
        grads_V = jac["grads_V"][0]   # (6, n_el)
        eps = 1e-5
        for v in range(6):
            xp = x0.copy(); xp[v] += eps
            xm = x0.copy(); xm[v] -= eps
            _, fVp = m.forward(xp, ec)
            _, fVm = m.forward(xm, ec)
            fd = (fVp[0] - fVm[0]) / (2 * eps)
            np.testing.assert_allclose(
                grads_V[v], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"ALM3D vol grad mismatch var={v}")
