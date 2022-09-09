import numpy as np
from scipy import interpolate
import toolbox as tb
from copy import copy
import bisect
from .integral_solvers import solve_quasi_dyson_adapt, cum_int_adapt_simpson

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
        bias_res=0.0,
        orbitals=[0],
        couplings=[1.0],
        eps_sys=0.0,
        eps_res=0.0,
        mu_res=0.0,
        D_res=1.0,
        eta_res=0.0,
        U=0.0,
    ):
        self.beta = beta
        self.bias_res = bias_res
        self.orbitals = orbitals
        self.couplings = couplings
        self.eps_sys = eps_sys
        self.eps_res = eps_res
        self.mu_res = mu_res
        self.D_res = D_res
        self.eta_res = eta_res
        self.U = U


class AccuracyParameters(Parameters):
    """
    Parameters for the accuracy of the calculation.

    Parameters:
        time_extrapolate -- Solve up to this time, extrapolate beyond
        tol_C -- tolerance in integrating dC/dt = V phi
        method -- method for quasi Dyson solver, one of "cheb", "trapz", "trapz-LU", "trapz-GMRES"
    """

    # TODO: update docstring

    def __init__(
        self,
        time_extrapolate,
        tol_C=1e-2,
        qdyson_rtol=1e-5,
        qdyson_atol=1e-5,
        method="trapz",
        tol_gmres=1e-10,
        atol_gmres=1e-10,
        qdyson_max_N=int(1e8),
        nr_freqs_res=int(1e4),
        w_max_res=100.0,
    ):
        self.time_extrapolate = time_extrapolate
        self.tol_C = tol_C
        self.qdyson_rtol = qdyson_rtol
        self.qdyson_atol = qdyson_atol
        self.method = method
        self.tol_gmres = tol_gmres
        self.atol_gmres = atol_gmres
        self.qdyson_max_N = qdyson_max_N
        self.nr_freqs_res = nr_freqs_res
        self.w_max_res = w_max_res


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
    params.qdyson_rtol *= 10.0
    yield params, "qdyson_rtol"

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

    def __init__(
        self, reservoir, orbitals, capacitive_couplings, accuracy_params, verbose=False
    ):
        """
        Arguments:
            reservoir -- a `Reservoir` instance
            orbitals -- list of reservoir orbitals where coupling acts
            capacitive_couplings -- list of corresponding couplings
            accuracy_params -- an instance of `AccuracyParameters`
        """
        self.reservoir = reservoir
        self.orbitals = np.asarray(orbitals)
        self.capacitive_couplings = np.asarray(capacitive_couplings)
        self.AP = copy(accuracy_params)
        self.verbose = verbose

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

    def A_plus_reta_w(self, Q, nr_freqs, freq_res_factor=0.1):
        """
        FT of A^+_Q(t) theta(t)

        Returns: freqs, A, energy shift.
        """
        type = 0
        self.compute_C(type, Q)

        intercept, slope = self._cache_C_tail[type][Q]

        times = np.linspace(0, 1.0 / (freq_res_factor * np.abs(slope.real)), nr_freqs)
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

        self._cached_phi = []
        self._times = []

        if self.verbose:
            print("[Xray/CorrelatorSolver] Computing C...", flush=True)

        times, C_vals, err = cum_int_adapt_simpson(
            lambda t: self.phi(sign, Q, t, use_cache_N=True, cache_N=True),
            self.AP.time_extrapolate,
            tol=self.AP.tol_C,
        )
        C_vals *= -sign

        if self.verbose:
            print("[Xray/CorrelatorSolver] ... Done.", flush=True)

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

    def phi(self, sign, Q, t, start_N=None, use_cache_N=False, cache_N=False):
        """
        Computes \phi_t(t, t^+) using the quasi Dyson equation.
        """
        # TODO: refactor calculation of phi (better N, individual interpolations, manage guesses maybe)
        assert t >= 0.0

        if np.abs(t) < 1e-10:
            out = 0.0j

            for orb, coupling in zip(self.orbitals, self.capacitive_couplings):
                out += coupling * self.reservoir.g_less_t_fun(Q)(orb, orb)(0.0)

            return out

        guess_flag = False
        if use_cache_N and len(self._times) > 0:
            idx = bisect.bisect(self._times, t)
            if idx != 0:
                _, start_N = self._cached_phi[idx - 1]
            if idx < len(self._times):
                t_cache = self._times[idx]
                all_guess_phi, _ = self._cached_phi[idx]
                guess_flag = True

        out = 0.0j
        N_max = 0
        all_phi_fun = []

        for j in range(len(self.orbitals)):

            if guess_flag:
                guess = lambda u: all_guess_phi[j](u + t_cache - t)
            else:
                guess = None

            phi_fun, err, N = solve_quasi_dyson_adapt(
                self.reservoir.g_less_t_fun(Q),
                self.reservoir.g_grea_t_fun(Q),
                t,
                self.orbitals[j],
                self.orbitals,
                -sign * self.capacitive_couplings,
                self.AP.qdyson_rtol,
                self.AP.qdyson_atol,
                start_N=start_N,
                guess=guess,
                method=self.AP.method,
                tol_gmres=self.AP.tol_gmres,
                atol_gmres=self.AP.atol_gmres,
                max_N=self.AP.qdyson_max_N,
            )

            all_phi_fun.append(phi_fun)
            N_max = max(N_max, N)

            out += self.capacitive_couplings[j] * phi_fun(t)[j]

        if cache_N:
            idx = bisect.bisect(self._times, t)
            self._times.insert(idx, t)
            self._cached_phi.insert(idx, (all_phi_fun, N_max // 2))

        return out

    def empty_qdyson_cache(self):
        self._times = []
        self._cached_phi = []

    ### getters ###
    def get_tail(self, type, Q):
        tail = self._cache_C_tail[type][Q]
        if tail is None:
            raise RuntimeError("Tail has not been computed.")

        return tail
