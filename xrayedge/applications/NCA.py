import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from copy import copy
import toolbox as tb
from ..solver import PhysicsParameters, AccuracyParameters, CorrelatorSolver
from ..reservoir import QPC

# TODO: rename
# TODO: test energy shifts


class NCASolver:
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

        self.correlator_solver = CorrelatorSolver(QPC(self.PP), self.PP.V_cap, self.AP)

    def G_grea(self, t_array):
        """
        Pseudo-particle greater Green function in times on the QD
        """
        # no U in NCA constraint
        return (
            -1j
            * np.exp(-1j * t_array * self.PP.eps_QD)
            * self.correlator_solver.A_plus(0, t_array)
        )

    def G_reta_w(self, nr_freqs):
        """
        Pseudo-particle greater-retarded Green function in frequencies on the QD.

        For NCA in the steady state regime, one only needs the greater quaisparticle GFs in the sector Q=0 (see notes).
        Also, the partition function is reduced to 1.

        Returns: freqs, G_grea, energy shift
        """
        # no U in NCA constraint
        w, A_w, energy_shift = self.correlator_solver.A_plus_reta_w(0, nr_freqs)
        return w, -1j * A_w, energy_shift + self.PP.eps_QD

    def G_reta_w_lorentzian_approx(self):
        """
        Assuming G_reta_w has a lorentzian shape 1 / (w - gamma),
        returns an estimate of gamma based on the long time behavior of C.
        """
        self.correlator_solver.compute_C(0, 0)
        tail = self.correlator_solver.get_tail(0, 0)
        slope = tail[1]
        return self.PP.eps_QD + 1j * slope


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


def interp_one_over_G_reta_w(w, wp, fp, wp_shift, rtol=1e-2, atol=1e-4, plot=False):

    wp_sh = wp + wp_shift

    mask_p = np.abs(wp) < wp[-1] / 10.0

    idx_left = np.nonzero(mask_p)[0][0]
    idx_right = np.nonzero(mask_p)[0][-1]
    mask_left = w <= wp_sh[mask_p][0]
    mask_right = w >= wp_sh[mask_p][-1]
    mask_center = ~np.logical_or(mask_left, mask_right)

    f_real = np.zeros_like(w)

    slope = (np.log(fp.real[idx_right]) - np.log(fp.real[idx_right - 1])) / (
        np.log(wp[idx_right]) - np.log(wp[idx_right - 1])
    )
    intercept = np.log(fp.real[idx_right]) - slope * np.log(wp[idx_right])
    f_real[mask_right] = np.exp(intercept + np.log(w[mask_right] - wp_shift) * slope)

    slope = (np.log(-fp.real[idx_left + 1]) - np.log(-fp.real[idx_left])) / (
        np.log(-wp[idx_left + 1]) - np.log(-wp[idx_left])
    )

    intercept = np.log(-fp.real[idx_left]) - slope * np.log(-wp[idx_left])
    f_real[mask_left] = -np.exp(intercept + np.log(-w[mask_left] + wp_shift) * slope)

    f_real[mask_center] = checked_interp(
        w[mask_center],
        wp_sh,
        fp.real,
        rtol=rtol,
        atol=atol,
        kind="cubic",
        assume_sorted=True,
        copy=False,
    )

    delta = abs(wp[-1] - wp[0]) * 0.1
    mask = np.logical_and(w >= wp_sh[0] + delta, w <= wp_sh[-1] - delta)

    assert mask.any()

    f_imag = np.empty_like(f_real)

    f_imag[mask] = checked_interp(
        w[mask],
        wp_sh,
        fp.imag,
        rtol=rtol,
        atol=atol,
        kind="cubic",
        assume_sorted=True,
        copy=False,
    )

    f_imag[~mask] = 0.0

    f = f_real + 1j * f_imag

    if plot:
        plt.plot(w, f.imag)
        plt.plot(wp_sh, fp.imag, "--")
        plt.semilogy()
        xcenter = (wp_sh[0] + wp_sh[-1]) / 2.0
        xdev = np.abs(wp_sh[0] - wp_sh[-1])
        plt.xlim(xcenter - xdev, xcenter + xdev)
        tb.autoscale_y(logscale=True)
        plt.show()

        plt.axvline(wp_shift + wp[-1] / 10.0, c="k", ls=":")
        plt.axvline(wp_shift - wp[-1] / 10.0, c="k", ls=":")
        plt.plot(w, f.real - w)
        plt.xlim(xcenter - xdev, xcenter + xdev)
        tb.autoscale_y()
        plt.plot(wp_sh, fp.real - wp_sh, "--")

        plt.show()

    norm_err = np.abs(-np.trapz(x=w, y=1.0 / f) - np.pi * 1j)
    if norm_err > 1e-1:
        print(f"XXX Norm F_reta_w error = {norm_err}")
    elif norm_err > 1e-2:
        print(f"/!\ Norm F_reta_w error = {norm_err}")

    return f
