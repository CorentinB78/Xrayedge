import numpy as np
from scipy import interpolate
from copy import copy
from .integral_solvers import solve_quasi_dyson_adapt, cum_int_adapt_simpson
import matplotlib
import matplotlib.pyplot as plt
import functools
import multiprocessing
from .fourier import fourier_transform


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
        qdyson_min_step=None,
        method="trapz",
        tol_gmres=1e-10,
        atol_gmres=1e-10,
        qdyson_max_N=int(1e8),
        nr_freqs_res=int(1e4),
        w_max_res=100.0,
        parallelize_orbitals=False,
    ):
        self.time_extrapolate = time_extrapolate
        self.tol_C = tol_C
        self.qdyson_rtol = qdyson_rtol
        self.qdyson_atol = qdyson_atol
        self.qdyson_min_step = qdyson_min_step
        self.method = method
        self.tol_gmres = tol_gmres
        self.atol_gmres = atol_gmres
        self.qdyson_max_N = qdyson_max_N
        self.nr_freqs_res = nr_freqs_res
        self.w_max_res = w_max_res
        self.parallelize_orbitals = parallelize_orbitals


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

    params = copy(accuracy_params)
    params.qdyson_atol *= 10.0
    yield params, "qdyson_atol"

    if gmres:
        params = copy(accuracy_params)
        params.tol_gmres *= 10.0
        yield params, "tol_gmres"

        params = copy(accuracy_params)
        params.atol_gmres *= 10.0
        yield params, "atol_gmres"


def _split_nr_ticks(n, j):
    q = 100 // n
    r = 100 % n
    return q + (1 if j < r else 0)


def _run(j, self, sign, Q):
    """
    Need this function to be defined at the top level of the module to be picklable.
    For use with multiprocessing pool in `CorrelatorSolver`.
    """
    nr_ticks = (
        _split_nr_ticks(len(self.orbitals), j) if self.AP.parallelize_orbitals else 100
    )

    f = functools.partial(self.phi, sign, Q, orb_idx=j)  # t -> phi(sign, Q, t, j)

    times, C_vals, err = cum_int_adapt_simpson(
        f,
        self.AP.time_extrapolate,
        tol=self.AP.tol_C,
        verbose=self.verbose,
        progress_nr_ticks=nr_ticks,
        multiproc_progress_bar=self.AP.parallelize_orbitals,
    )

    return times, C_vals, err


def symmetrize(coord, values, center, function=None, axis=-1, snap=1e-10):
    coord_out = np.asarray(coord).copy()
    values_out = np.moveaxis(np.asarray(values).copy(), axis, -1)

    if len(coord_out) != values_out.shape[-1]:
        raise ValueError("This axis does not match the number of coordinates!")

    snap = np.abs(snap)
    s = slice(None, None, -1)
    if function is None:
        function = lambda x: x
    if coord_out[0] + snap < center < coord_out[-1] - snap:
        raise ValueError("center is within the coordinate range")
    elif center <= coord_out[0] + snap:
        if np.abs(center - coord_out[0]) <= snap:
            s = slice(None, 0, -1)

        coord_out = np.concatenate((-coord_out[s] + 2 * center, coord_out))
        values_out = np.concatenate((function(values_out[..., s]), values_out), axis=-1)

    elif center >= coord_out[-1] - snap:
        if np.abs(center - coord_out[-1]) <= snap:
            s = slice(-2, None, -1)

        coord_out = np.concatenate((coord_out, -coord_out[s] + 2 * center))
        values_out = np.concatenate((values_out, function(values_out[..., s])), axis=-1)

    return coord_out, np.moveaxis(values_out, -1, axis)


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
        self.nr_iter_dict = {}

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
        w, A_w = fourier_transform(times, A_bulk)

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

    def compute_C(self, type, Q, force_recompute=False):
        """
        If value cannot be found in cache, does compute C, fills cache and returns error estimate.
        """

        if (not force_recompute) and (self._cache_C_interp[type][Q] is not None):
            return None  # nothing to do

        if type == 0:
            sign = 1
        elif type == 1:
            sign = -1
        else:
            raise ValueError

        self.nr_iter_dict = {}
        C_interp_list = []
        slope = 0.0j
        intercept = 0.0j
        err_tot = 0.0

        run = functools.partial(_run, sign=sign, Q=Q, self=self)

        # TODO: use threadpoolctl to limit number of threads automatically
        # TODO: find a way for cached reservoir data to survive inside multiprocessing

        if self.AP.parallelize_orbitals:
            if self.verbose:
                print(
                    "Simpson progress (out of 100): - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |",
                    flush=True,
                )

            with multiprocessing.Pool(len(self.orbitals)) as p:
                res = p.map(run, range(len(self.orbitals)))

            if self.verbose:
                print()

        else:
            res = map(run, range(len(self.orbitals)))

        for times, C_vals, err in res:

            C_vals *= -sign
            err_tot += err

            C_interp_list.append(
                interpolate.CubicSpline(
                    *symmetrize(times, C_vals, 0.0, lambda x: np.conj(x)),
                    bc_type="natural",
                    extrapolate=False,
                )
            )

            slope_j = C_interp_list[-1].derivative()(times[-1])
            slope += slope_j
            intercept += C_vals[-1] - slope_j * times[-1]

        self._cache_C_interp[type][Q] = lambda u: np.sum(
            [f(u) for f in C_interp_list], axis=0
        )
        self._cache_C_tail[type][Q] = (intercept, slope)

        return err_tot

    def phi(self, sign, Q, t, orb_idx):
        """
        Computes \phi_t(t, t^+) using the quasi Dyson equation.
        """
        assert t >= 0.0
        orb = self.orbitals[orb_idx]
        coupling = self.capacitive_couplings[orb_idx]

        if np.abs(t) < 1e-10:
            return coupling * self.reservoir.g_less_t_fun(Q)(orb, orb)(0.0)

        if self.AP.qdyson_min_step is None:
            start_N = None
        else:
            start_N = 2 ** round(np.log2(t / self.AP.qdyson_min_step))
            start_N = max(16, start_N)

        phi_fun, err, nr_iter_dict = solve_quasi_dyson_adapt(
            self.reservoir.g_less_t_fun(Q),
            self.reservoir.g_grea_t_fun(Q),
            t,
            orb,
            self.orbitals,
            -sign * self.capacitive_couplings,
            self.AP.qdyson_rtol,
            self.AP.qdyson_atol,
            start_N=start_N,
            method=self.AP.method,
            tol_gmres=self.AP.tol_gmres,
            atol_gmres=self.AP.atol_gmres,
            max_N=self.AP.qdyson_max_N,
        )

        self.nr_iter_dict[(t, orb_idx)] = nr_iter_dict

        return coupling * phi_fun(t)[orb_idx]

    ### getters ###
    def get_tail(self, type, Q):
        tail = self._cache_C_tail[type][Q]
        if tail is None:
            raise RuntimeError("Tail has not been computed.")

        return tail

    ### convenience ###
    def plot_nr_GMRES_iter(self):
        cmap = matplotlib.cm.get_cmap("Spectral")

        for key, dict2 in self.nr_iter_dict.items():
            t, j = key
            plt.plot(
                dict2.keys(), dict2.values(), "o-", c=cmap(t / self.AP.time_extrapolate)
            )

        plt.xlabel("matrix size")
        plt.ylabel("nr GMRES iterations")
        plt.semilogx()
        plt.show()
