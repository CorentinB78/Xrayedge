import numpy as np
from scipy import interpolate
import toolbox as tb
from copy import copy
from .integral_solvers import solve_quasi_dyson, cum_semiinf_adpat_simpson

# TODO parallelize?
# TODO cleanup notes!


############## Model ###############


class Parameters:
    def __repr__(self):
        out = "Parameters object containing:\n"
        for key, val in self.__dict__.items():
            out += f"{key} = {val}\n"

        return out


# TODO: Reorganize parameters. Current ones are specialized for QPC.


class PhysicsParameters(Parameters):
    """
    Parameters of the Hamiltonian and statistics.
    """

    def __init__(
        self,
        beta=1.0,
        mu_sys=0.0,
        bias_res=0.0,
        V_cap=1.0,
        eps_sys=0.0,
        eps_res=0.0,
        D_res=1.0,
        eta_res=0.01,
        U=0.0,
    ):
        self.beta = beta
        self.mu_sys = mu_sys  # chemical potential on the QD TODO: remove if redondant with eps_sys
        self.bias_res = bias_res
        self.V_cap = V_cap  # = dV/dQ
        self.eps_sys = eps_sys  # on the QD
        self.eps_res = eps_res  # on the QPC
        self.D_res = D_res
        self.eta_res = eta_res
        self.U = U


class AccuracyParameters(Parameters):
    """
    Parameters for the accuracy of the calculation.

    Parameters:
        time_extrapolate -- Solve up to this time, extrapolate beyond
        tol_C -- tolerance in integrating dC/dt = V phi
        delta_interp_phi -- time step in quasi Dyson solver to compute phi. Should be smaller than timescale of variation of bare g.
        method -- method for quasi Dyson solver, one of "cheb", "trapz", "trapz-LU", "trapz-GMRES"
    """

    def __init__(
        self,
        time_extrapolate,
        tol_C=1e-2,
        delta_interp_phi=0.05,
        method="trapz",
        tol_gmres=1e-5,
        atol_gmres=1e-5,
    ):
        self.time_extrapolate = time_extrapolate
        self.tol_C = tol_C
        self.delta_interp_phi = delta_interp_phi
        self.method = method
        self.tol_gmres = tol_gmres
        self.atol_gmres = atol_gmres

    def nr_pts_phi(self, t):
        if self.method == "cheb":
            return int(np.pi * np.sqrt(t / (2 * self.delta_interp_phi)) + 3)

        ### default to linear grid
        return int(t / self.delta_interp_phi + 3)


def gen_params(accuracy_params, gmres=False):
    """
    Generator yielding variations of accuracy parameters for convergence checks.

    Yield (ap, label)
    """

    yield copy(accuracy_params), "original"

    params = copy(accuracy_params)
    params.time_extrapolate /= 2.0
    yield params, "time_extrapolate"

    params = copy(accuracy_params)
    params.tol_C *= 10.0
    yield params, "tol_C"

    params = copy(accuracy_params)
    params.delta_interp_phi *= 2.0
    yield params, "delta_interp_phi"

    if gmres:
        params = copy(accuracy_params)
        params.tol_gmres *= 10.0
        yield params, "tol_gmres"

        params = copy(accuracy_params)
        params.atol_gmres *= 10.0
        yield params, "atol_gmres"


class CorrelatorSolver:
    """
    Solver for computing correlation functions in a finite system capacitively coupled to a reservoir.

    Data is cached after calculation, so parameters should not be changed.
    """

    def __init__(self, reservoir, capacitive_coupling, accuracy_params):
        self.reservoir = reservoir
        self.V = capacitive_coupling
        self.AP = copy(accuracy_params)

        self.N = 3  # nr of different charge states affecting the QPC
        self._cache_C_interp = [[None] * self.N, [None] * self.N]
        self._cache_C_tail = [[None] * self.N, [None] * self.N]

    def A_plus(self, Q, times):
        """
        A^+_Q(t)
        """
        return np.exp(self.C(0, Q, times))

    def A_minus(self, Q, times):
        """
        A^-_Q(t)
        """
        return np.exp(self.C(1, Q, times))

    def A_plus_reta_w(self, Q, nr_freqs):
        """
        FT of A^+_Q(t) theta(t)

        Returns: freqs, A, energy shift.
        """
        type = 0
        self.compute_C(type, Q)

        intercept, slope = self._cache_C_tail[type][Q]

        times = np.linspace(0, 10.0 / np.abs(slope.real), nr_freqs)
        C_vals = self.C(type, Q, times)

        # shift energy
        C_vals -= 1j * times * slope.imag

        A_bulk = np.exp(C_vals)

        # treat tail analytically
        A_bulk -= np.exp(intercept + times * slope.real)

        A_bulk[0] *= 0.5
        w, A_w = tb.fourier_transform(times, A_bulk)

        A_w += -np.exp(intercept) / (1j * w + slope.real)

        return w, A_w, -slope.imag

    ######## C and phi #######

    def C(self, type, Q, times):
        """
        Returns values of C on coordinates `times`.

        Values beyond the extrapolation time are obtained by a linear extrapolation.

        Arguments:
            type -- 0 or 1, resp. for A^+ or A^- process
            Q -- integer, occupation of QD
            times -- 1D array

        Returns:
            C_vals -- a 1D array
        """
        times = np.asarray(times)
        self.compute_C(type, Q)

        C_vals = self._cache_C_interp[type][Q](times)
        mask = np.abs(times) >= self.AP.time_extrapolate

        if mask.any():
            intercept, slope = self._cache_C_tail[type][Q]
            tt = times[mask]
            C_vals[mask] = (
                intercept.real
                + np.abs(tt) * slope.real
                + 1j * (np.sign(tt) * intercept.imag + tt * slope.imag)
            )

        return C_vals

    def compute_C(self, type, Q, ignore_cache=False):
        """
        If value cannot be found in cache, does compute C, fills cache and returns error estimate.
        """

        if (not ignore_cache) and (self._cache_C_interp[type][Q] is not None):
            return None  # nothing to do

        if type == 0:
            sign = 1
        elif type == 1:
            sign = -1
        else:
            raise ValueError

        times, C_vals, err = cum_semiinf_adpat_simpson(
            lambda t: self.phi(sign, Q, t),
            scale=self.AP.time_extrapolate,
            tol=self.AP.tol_C,
            extend=False,
        )
        C_vals *= -sign * self.V
        err *= np.abs(self.V)

        slope = (C_vals[-1] - C_vals[-2]) / (times[-1] - times[-2])
        intercept = C_vals[-1] - slope * times[-1]

        C_interp = interpolate.CubicSpline(
            *tb.symmetrize(times, C_vals, 0.0, lambda x: np.conj(x)),
            bc_type="natural",
            extrapolate=False,
        )
        self._cache_C_interp[type][Q] = C_interp
        self._cache_C_tail[type][Q] = (intercept, slope)

        return err

    def phi(self, sign, Q, t):
        """
        Computes \phi_t(t, t^+) using the quasi Dyson equation.
        """
        assert t >= 0.0
        if np.abs(t) < 1e-10:
            return self.reservoir.g_less_t_fun(Q)(0.0)

        times, phi = solve_quasi_dyson(
            self.reservoir.g_less_t_fun(Q),
            self.reservoir.g_grea_t_fun(Q),
            t,
            -sign * self.V,
            self.AP.nr_pts_phi(t),
            method=self.AP.method,
            tol_gmres=self.AP.tol_gmres,
            atol_gmres=self.AP.atol_gmres,
        )
        return phi[-1]

    ### getters ###
    def get_tail(self, type, Q):
        tail = self._cache_C_tail[type][Q]
        if tail is None:
            raise RuntimeError("Tail has not been computed.")

        return tail
