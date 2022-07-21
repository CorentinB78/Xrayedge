import numpy as np
from scipy import interpolate, integrate
from matplotlib import pyplot as plt
from copy import copy
import toolbox as tb
from ..solver import PhysicsParameters, AccuracyParameters, CorrelatorSolver
from ..reservoir import QuantumDot
import bisect

# TODO: rename
# TODO: test energy shifts


class XrayForNCASolver:
    """
    Solver for computing Pseudo-particle Green functions of an Anderson impurity capacitively coupled to a QPC.
    """

    def __init__(self, physics_params=None, accuracy_params=None):
        self.PP = (
            copy(physics_params) if physics_params is not None else PhysicsParameters()
        )
        self.AP = (
            copy(accuracy_params)
            if accuracy_params is not None
            else AccuracyParameters(1.0)
        )

        self.correlator_solver = CorrelatorSolver(
            QuantumDot(self.PP), self.PP.V_cap, self.AP
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


def checked_interp(x, xp, fp, rtol, atol, **kwargs):
    """
    Assumed x regularly spaced
    """
    interp = interpolate.interp1d(xp, fp, **kwargs)
    f = interp(x)
    f_half = interp(x[::2])
    f_half = np.interp(x, x[::2], f_half)

    diff = np.abs(f_half - f)
    err = np.max(diff - rtol * np.abs(f) - atol)

    if err > 0:
        print(
            f"XXX [Xray] low number of samples for interp: max abs err={np.max(diff)}, max rel err={np.max(diff / np.abs(f))}"
        )

    return f


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


def extrap_G_reta_w(x, xp, yp, tol, rtol=1e-2, atol=1e-4, plot=False):

    ### real part
    x0 = xp[-1] / 10.0
    assert x0 > 0.0
    f_real = np.empty_like(x)
    mask = x > x0
    f_real[mask] = tangent_power_law(xp, yp.real, x0=x0)(x[mask])
    mask = x < -x0
    f_real[mask] = tangent_power_law(xp, yp.real, x0=-x0)(x[mask])
    mask = np.abs(x) <= x0
    f_real[mask] = checked_interp(
        x[mask],
        xp,
        yp.real,
        rtol=rtol,
        atol=atol,
        kind="cubic",
        assume_sorted=True,
        copy=False,
    )

    if plot:
        plt.plot(x, f_real, "--")
        plt.plot(xp, yp.real)

        plt.loglog()
        tb.autoscale_y(logscale=True)

        plt.axvline(x0, c="k", ls=":", alpha=0.4)
        plt.show()

        plt.plot(-x, -f_real, "--")
        plt.plot(-xp, -yp.real)

        plt.loglog()
        tb.autoscale_y(logscale=True)

        plt.axvline(x0, c="k", ls=":", alpha=0.4)
        plt.show()

    ### imag part
    mask = -yp.imag > tol

    idx_left = np.nonzero(mask)[0][0]
    idx_right = np.nonzero(mask)[0][-1]

    f_imag = np.empty_like(x)
    mask = x < xp[idx_left]
    f_imag[mask] = tangent_exp(xp, yp.imag, idx=idx_left)(x[mask])
    mask = x > xp[idx_right]
    f_imag[mask] = tangent_power_law(xp, yp.imag, idx=idx_right)(x[mask])
    mask = np.logical_and(x >= xp[idx_left], x <= xp[idx_right])
    f_imag[mask] = checked_interp(
        x[mask],
        xp,
        yp.imag,
        rtol=rtol,
        atol=atol,
        kind="cubic",
        assume_sorted=True,
        copy=False,
    )

    if plot:
        plt.plot(xp, -yp.imag)
        plt.xlim(*plt.xlim())  # freeze xlim
        plt.semilogy()
        tb.autoscale_y(logscale=True)

        plt.plot(x, -f_imag, "--")

        plt.axhline(tol, c="k", ls=":", alpha=0.4)
        plt.axvline(xp[idx_left], c="k", ls=":", alpha=0.4)
        plt.axvline(xp[idx_right], c="k", ls=":", alpha=0.4)

        plt.show()

    f = f_real + 1j * f_imag

    ### check norm

    norm_err = np.abs(-integrate.simpson(x=x, y=f) - np.pi * 1j)
    if norm_err > 1e-1:
        print(f"XXX Norm F_reta_w = {integrate.simpson(x=x, y=f)}")
    elif norm_err > 1e-2:
        print(f"/!\ Norm F_reta_w = {integrate.simpson(x=x, y=f)}")

    ### plot inverse

    if plot:
        plt.plot(xp, np.imag(1.0 / yp))

        plt.xlim(*plt.xlim())  # freeze xlim
        plt.semilogy()
        tb.autoscale_y(logscale=True)

        plt.plot(x, np.imag(1.0 / f), "--")

        plt.show()

    return f
