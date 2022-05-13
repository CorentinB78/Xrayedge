import numpy as np
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


def clean_and_interp_G_reta_w(w, wp, fp, wp_shift, tol=1e-3, plot=False):

    wp_sh = wp + wp_shift

    f_real = 1.0 / (w - wp_shift)
    mask = np.abs(w - wp_shift) < wp[-1] / 10.0
    f_real[mask] = np.interp(w[mask], wp_sh, fp.real)

    mask_p = -fp.imag > tol

    idx_left = np.nonzero(mask_p)[0][0]
    idx_right = np.nonzero(mask_p)[0][-1]
    mask_left = w <= wp_sh[mask_p][0]
    mask_right = w >= wp_sh[mask_p][-1]
    mask_center = ~np.logical_or(mask_left, mask_right)

    f_imag = np.zeros_like(w)

    slope = (np.log(-fp.imag[idx_left + 1]) - np.log(-fp.imag[idx_left])) / (
        wp_sh[idx_left + 1] - wp_sh[idx_left]
    )

    if slope < 0.0:
        print(f"XXX slope = {slope} is negative!")

    intercept = np.log(-fp.imag[idx_left]) - slope * (wp_sh[idx_left])
    f_imag[mask_left] = -np.exp(intercept + w[mask_left] * slope)

    f_imag[mask_center] = np.interp(w[mask_center], wp_sh, fp.imag)

    slope = (np.log(-fp.imag[idx_right]) - np.log(-fp.imag[idx_right - 1])) / (
        np.log(wp[idx_right]) - np.log(wp[idx_right - 1])
    )
    intercept = np.log(-fp.imag[idx_right]) - slope * np.log(wp[idx_right])
    f_imag[mask_right] = -np.exp(intercept + np.log(w[mask_right] - wp_shift) * slope)

    f = f_real + 1j * f_imag

    if plot:
        plt.axhline(tol, c="k", ls=":")
        plt.plot(w, -f.imag)
        plt.plot(wp_sh, -fp.imag, "--")
        plt.semilogy()
        xcenter = (wp_sh[0] + wp_sh[-1]) / 2.0
        xdev = np.abs(wp_sh[0] - wp_sh[-1])
        plt.xlim(xcenter - xdev, xcenter + xdev)
        tb.autoscale_y(logscale=True)
        tb.ylim_max(tol * 1e-3, 1e10 * tol)
        # plt.ylim(tol * 1e-3)
        plt.show()

        plt.axvline(wp_shift + wp[-1] / 10.0, c="k", ls=":")
        plt.axvline(wp_shift - wp[-1] / 10.0, c="k", ls=":")
        plt.plot(w, np.abs(f.real))
        plt.plot(wp_sh, np.abs(fp.real), "--")
        plt.semilogy()

        plt.xlim(xcenter - xdev, xcenter + xdev)
        tb.autoscale_y(logscale=True)
        plt.show()

    norm_err = np.abs(-np.trapz(x=w, y=f) - np.pi * 1j)
    if norm_err > 1e-1:
        print(f"XXX Norm F_reta_w error = {norm_err}")
    elif norm_err > 1e-2:
        print(f"/!\ Norm F_reta_w error = {norm_err}")
    # print("Norm F_reta_w:", -np.trapz(x=w, y=f) / np.pi, "== 1.0j ?")

    return f
