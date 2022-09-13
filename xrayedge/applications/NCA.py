import numpy as np
from scipy import interpolate, integrate
from matplotlib import pyplot as plt
from copy import copy
import toolbox as tb
from ..solver import CorrelatorSolver
import bisect

# TODO: test energy shifts


class XrayForNCASolver:
    """
    Solver for computing Pseudo-particle Green functions of an Anderson impurity capacitively coupled to a reservoir.
    """

    def __init__(self, reservoir, physics_params, accuracy_params):
        self.PP = copy(physics_params)
        self.AP = copy(accuracy_params)

        self.correlator_solver = CorrelatorSolver(
            reservoir,
            self.PP.orbitals,
            self.PP.couplings,
            self.AP,
        )

    def G_grea(self, t_array):
        """
        Pseudo-particle greater Green function in times on the QD
        """
        # no U in NCA constraint
        return (
            -1j
            * np.exp(-1j * t_array * self.PP.eps_sys)
            * self.correlator_solver.A_plus(0, t_array)
        )

    def G_reta_w(self, nr_freqs, freq_res_factor=0.1):
        """
        Pseudo-particle greater-retarded Green function in frequencies on the QD.

        For NCA in the steady state regime, one only needs the greater quasiparticle GFs in the sector Q=0 (see notes).
        Also, the partition function is reduced to 1.

        Returns: freqs, G_grea, energy shift
        """
        # no U in NCA constraint
        w, A_w, energy_shift = self.correlator_solver.A_plus_reta_w(
            0, nr_freqs, freq_res_factor=freq_res_factor
        )
        return w, -1j * A_w, energy_shift + self.PP.eps_sys

    def G_reta_w_lorentzian_approx(self):
        """
        Assuming G_reta_w has a lorentzian shape 1 / (w - gamma),
        returns an estimate of gamma based on the long time behavior of C.
        """
        self.correlator_solver.compute_C(0, 0)
        tail = self.correlator_solver.get_tail(0, 0)
        slope = tail[1]
        return self.PP.eps_sys + 1j * slope


def tangent_exp(x, y, x0=None, idx=None):
    """
    Return function of x of the form a.e^{bx}, so that it is tangent to y(x) at x=x0.

    Assumes x sorted and y real
    """
    if idx is None:
        assert x0 is not None
        idx = bisect.bisect(x, x0)
    else:
        assert x0 is None
        x0 = x[idx]

    if idx == 0:
        raise ValueError("x0 should be > x[1]")
    if idx == len(x) - 1:
        raise ValueError("x0 should be < x[-2]")
    f0 = y[idx]

    if abs(f0) < 1e-200:
        # then we assume Df0 is also zero
        return lambda t: 0.0

    Df0 = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])

    b = Df0 / f0

    return lambda t: f0 * np.exp(b * (t - x0))


def tangent_power_law(x, y, x0=None, idx=None):
    """
    Return function of x of the form a.x^b, so that it is tangent to y(x) at x=x0.

    Assumes x sorted and y real and y(x0) != 0
    """
    if idx is None:
        assert x0 is not None
        idx = bisect.bisect(x, x0)
    else:
        assert x0 is None
        x0 = x[idx]

    if idx < 1:
        raise ValueError("x0 should be > x[1]")
    if idx > len(x) - 2:
        raise ValueError("x0 should be < x[-2]")
    f0 = y[idx]
    Df0 = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])

    b = x0 * Df0 / f0

    return lambda t: f0 * (t / x0) ** b


class GFWithTails:
    def __init__(self, omegas, gf_vals, en_shift):
        self._omegas = omegas
        self._gf_vals = gf_vals
        self._en_shift = en_shift

        self._central_interp = interpolate.CubicSpline(
            omegas,
            gf_vals,
            bc_type="natural",
            extrapolate=False,
        )
        self._real_bounds = [None, None]
        self._imag_bounds = [None, None]
        self._pos_real_tail = None  # callable for the real part of the w > 0 tail
        self._pos_imag_tail = None  # callable for the imag part of the w > 0 tail
        self._neg_real_tail = None  # callable for the real part of the w < 0 tail
        self._neg_imag_tail = None  # callable for the imag part of the w < 0 tail

        self._vec_call = np.vectorize(self._call)

    def extrap_tails(self, tol):
        self._tol = tol
        xp = self._omegas
        yp = self._gf_vals

        ### real part
        x0 = xp[-1] / 10.0
        self._real_bounds = [-x0, x0]
        assert x0 > 0.0
        self._pos_real_tail = tangent_power_law(xp, yp.real, x0=x0)
        self._neg_real_tail = tangent_power_law(xp, yp.real, x0=-x0)

        ### imag part
        mask = -yp.imag > tol

        idx_left = np.nonzero(mask)[0][0]
        idx_right = np.nonzero(mask)[0][-1]

        self._imag_bounds = [xp[idx_left], xp[idx_right]]

        self._neg_imag_tail = tangent_exp(xp, yp.imag, idx=idx_left)
        self._pos_imag_tail = tangent_power_law(xp, yp.imag, idx=idx_right)

    def check_norm(self):
        norm_re, err_re = integrate.quad(
            lambda w: self._call(w + self._en_shift).real
            + self._call(-w + self._en_shift).real,
            0.0,
            np.inf,
        )
        norm_im, err_im = integrate.quad(
            lambda w: self._call(w + self._en_shift).imag
            + self._call(-w + self._en_shift).imag,
            0.0,
            np.inf,
        )
        norm = -(norm_re + 1j * norm_im) / np.pi

        norm_err = np.abs(norm - 1.0j)
        if norm_err > 1e-1:
            print(f"XXX Norm F_reta_w = {norm} != 1j")
        elif norm_err > 1e-2:
            print(f"/!\ Norm F_reta_w = {norm} != 1j")

    def _call(self, omega):
        omega -= self._en_shift
        if omega < self._real_bounds[0]:
            f_real = self._neg_real_tail(omega)
        elif omega <= self._real_bounds[1]:
            f_real = self._central_interp(omega).real
        else:
            f_real = self._pos_real_tail(omega)

        if omega < self._imag_bounds[0]:
            f_imag = self._neg_imag_tail(omega)
        elif omega <= self._imag_bounds[1]:
            f_imag = self._central_interp(omega).imag
        else:
            f_imag = self._pos_imag_tail(omega)

        return f_real + 1.0j * f_imag

    def __call__(self, omega):
        return self._vec_call(omega)

    def plot(self, omegas=None, filename=None, real_part=False, inverse=False):
        if omegas is None:
            x = np.linspace(
                -2 * abs(self._imag_bounds[0]), 2 * abs(self._imag_bounds[1]), 1000
            )
        else:
            x = omegas

        f = self(x + self._en_shift)  # unshift first

        x0, f0 = tb.vcut(
            self._omegas,
            self._gf_vals,
            -2 * abs(self._imag_bounds[0]),
            2 * abs(self._imag_bounds[1]),
        )

        # real part
        if real_part:

            plt.plot(x, f.real, "--")
            plt.plot(x0, f0.real)

            plt.loglog()
            tb.autoscale_y(logscale=True)

            plt.axvline(self._real_bounds[1], c="k", ls=":", alpha=0.4)
            plt.show()

            plt.plot(-x, -f.real, "--")
            plt.plot(-x0, -f0.real)

            plt.loglog()
            tb.autoscale_y(logscale=True)

            plt.axvline(-self._real_bounds[0], c="k", ls=":", alpha=0.4)
            plt.show()

        # imag part
        plt.plot(x0, -f0.imag, label="original data")
        # plt.xlim(*plt.xlim())  # freeze xlim
        plt.semilogy()
        # tb.autoscale_y(logscale=True)

        plt.plot(x, -f.imag, "--", label="extrapolation")

        plt.axhline(self._tol, c="k", ls=":", alpha=0.4)
        plt.axvline(self._imag_bounds[0], c="k", ls=":", alpha=0.4)
        plt.axvline(self._imag_bounds[1], c="k", ls=":", alpha=0.4)

        plt.legend()

        if filename is not None:
            plt.savefig(filename)
        plt.show()

        ### plot inverse
        if inverse:

            plt.plot(x0, np.imag(1.0 / f0))

            # plt.xlim(*plt.xlim())  # freeze xlim
            plt.semilogy()
            # tb.autoscale_y(logscale=True)

            plt.plot(x, np.imag(1.0 / f), "--")

            plt.show()
