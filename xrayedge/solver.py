import numpy as np
from scipy import interpolate
import toolbox as tb
from copy import copy
from .integral_solvers import solve_pseudo_dyson, cum_semiinf_adpat_simpson
from .reservoir import Reservoir, QPC

# TODO write test against asymptotic result
# TODO parallelize?
# TODO cleanup notes!


############## Model ###############


class PhysicsParameters:
    """
    Parameters of the Hamiltonian and statistics.
    """

    def __init__(
        self,
        beta=1.0,
        mu_d=0.0,
        bias=0.0,
        V_cap=1.0,
        eps_d=0.0,
        eps_c=0.0,
        mu_c=0.0,
        Gamma=1.0,
        U=0.0,
    ):
        self.beta = beta
        self.mu_d = (
            mu_d  # chemical potential on the QD TODO: remove if redondant with eps_d
        )
        self.bias = bias
        self.capac_inv = V_cap  # = dV/dQ
        self.eps_d = eps_d  # on the QD
        self.eps_c = eps_c  # on the QPC
        self.mu_c = mu_c
        self.Gamma = Gamma
        self.U = U

        # nr_channels = 4
        # lambda_phi = 1. # = capac ?
        # lambda_chi = 0. # = capac ?
        # ksi_0 = 10.

        # def delta_phi(self):
        #     return np.arctan(np.pi * self.lambda_phi * self.Fermi_dos())

        # def Fermi_dos(self):
        #     return self.Gamma / (self.eps_d**2 + self.Gamma**2) / np.pi


class AccuracyParameters:
    """
    Parameters for the accuracy of the calculation.
    Some are derived from the physics parameters.

    methods available: trapz, cheb
    fft_w_max is in unit of Gamma
    delta_interp_phi: reduce to increase precision. Should be smaller than timescale of variation of bare g.
    """

    def __init__(
        self,
        physics_params,
        time_extrapolate,
        fft_nr_samples=50000,
        fft_w_max=50.0,
        tol_C=1e-2,
        delta_interp_phi=0.05,
        method="trapz",
    ):
        self.PP = copy(physics_params)

        self.time_extrapolate = time_extrapolate
        self.fft_nr_samples = fft_nr_samples
        self.fft_w_max = fft_w_max  # in unit of Gamma
        self.tol_C = tol_C
        self.delta_interp_phi = delta_interp_phi
        self.method = method

    def omegas_fft(self):
        w_max = self.fft_w_max * self.PP.Gamma
        return np.linspace(
            -w_max, w_max, tb.misc._next_regular(int(self.fft_nr_samples))
        )

    def nr_pts_phi(self, t):
        if self.method == "cheb":
            return int(np.pi * np.sqrt(t / (2 * self.delta_interp_phi)) + 3)

        ### default to linear grid
        return int(t / self.delta_interp_phi + 3)


def gen_params(accuracy_params):
    """
    Generator yielding variations of accuracy parameters for convergence checks.

    Yield (ap, label)
    """

    yield copy(accuracy_params), "original"

    params = copy(accuracy_params)
    params.time_extrapolate /= 2.0
    yield params, "time_extrapolate"

    params = copy(accuracy_params)
    params.tol_C *= 4.0
    yield params, "tol_C"

    params = copy(accuracy_params)
    params.delta_interp_phi *= 2.0
    yield params, "delta_interp_phi"

    params = copy(accuracy_params)
    params.fft_nr_samples /= 2
    yield params, "fft_nr_samples"

    params = copy(accuracy_params)
    params.fft_w_max /= 2.0
    params.fft_nr_samples /= 2
    yield params, "fft_w_max"


class GFModel:
    """
    Base model for Green function calculation.
    Only contains basic relations.
    """

    def __init__(self, physics_params=None, accuracy_params=None):
        self.PP = (
            copy(physics_params) if physics_params is not None else PhysicsParameters()
        )
        self.AP = (
            copy(accuracy_params)
            if accuracy_params is not None
            else AccuracyParameters(self.PP, 1.0)
        )

    def weight(self, Q_up, Q_dn):
        return np.exp(
            -self.PP.beta
            * ((self.PP.eps_d - self.PP.mu_d) * (Q_up + Q_dn) + Q_up * Q_dn * self.PP.U)
        )

    def Z_d(self):
        """
        Partition function
        """
        return (
            self.weight(0, 0)
            + self.weight(1, 0)
            + self.weight(0, 1)
            + self.weight(1, 1)
        )

    def proba(self, Q_up, Q_dn):
        return self.weight(Q_up, Q_dn) / self.Z_d()

    def A_plus(self, Q, t_array):
        raise NotImplementedError

    def A_plus_w(self, Q, nr_freqs):
        raise NotImplementedError

    def A_minus(self, Q, t_array):
        raise NotImplementedError

    def G_grea(self, t_array):
        """
        Greater Green function in times on the QD
        """
        prefactor = -1j * np.exp(-1j * t_array * self.PP.eps_d)
        out = self.proba(0, 0) * self.A_plus(0, t_array)
        out += (
            np.exp(-1j * t_array * self.PP.U)
            * self.proba(0, 1)
            * self.A_plus(1, t_array)
        )
        return prefactor * out

    def G_less(self, t_array):
        """
        Lesser Green function in times on the QD
        """
        prefactor = 1j * np.exp(-1j * t_array * self.PP.eps_d)
        out = self.proba(1, 0) * np.conj(self.A_minus(1, t_array))
        out += (
            np.exp(-1j * t_array * self.PP.U)
            * self.proba(1, 1)
            * np.conj(self.A_minus(2, t_array))
        )
        return prefactor * out

    def G_grea_NCA_constraint(self, t_array):
        """
        Greater Green function in times on the QD under NCA constrain.
        """
        # no U in NCA constraint
        return (
            -1j
            * np.exp(-1j * t_array * self.PP.eps_d)
            * self.weight(0, 0)
            * self.A_plus(0, t_array)
        )

    def G_reta_w_NCA_constraint(self, nr_freqs):
        """
        Greater-retarded Green function in frequencies on the QD under NCA constrain.

        For NCA in the steady state regime, one only needs the greater quaisparticle GFs in the sector Q=0 (see notes).
        Also, the partition function is reduced to 1.

        Returns: freqs, G_grea, energy shift
        """
        # no U in NCA constraint
        w, A_w, energy_shift = self.A_plus_reta_w(0, nr_freqs)
        return w, -1j * self.weight(0, 0) * A_w, energy_shift - self.PP.eps_d


class NumericModel(GFModel):
    """
    Model for numerical calculation of correlators and Green functions.

    Data is cached after calculation, so parameters should not be changed.
    """

    def __init__(self, *args, **kwargs):
        super(NumericModel, self).__init__(*args, **kwargs)
        self.N = 3  # nr of different charge states affecting the QPC
        self._cache_C_interp = [[None] * self.N, [None] * self.N]
        self._cache_C_tail = [[None] * self.N, [None] * self.N]

        self.reservoir = QPC(self.PP, self.AP)

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
        if self._cache_C_tail[type][Q] is None:
            self.compute_C(type, Q)

        slope = self._cache_C_tail[type][Q][1]

        times = np.linspace(0, 200.0 / np.abs(slope.real), nr_freqs)
        C_vals = self.C(type, Q, times)

        # shift energy
        C_vals -= 1j * times * slope.imag

        A = np.exp(C_vals)
        A[0] *= 0.5
        w, A = tb.fourier_transform(times, A)

        return w, A, -slope.imag

    ######## C and phi #######

    def C(self, type, Q, times):
        times = np.asarray(times)
        if self._cache_C_interp[type][Q] is None:
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

    def compute_C(self, type, Q):
        """
        Compute C, fills cache and returns error estimate.
        """
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
        C_vals *= sign * self.PP.capac_inv
        err *= np.abs(self.PP.capac_inv)

        slope = (C_vals[-1] - C_vals[-2]) / (times[-1] - times[-2])
        intercept = C_vals[-1] - slope * times[-1]

        C_interp = interpolate.CubicSpline(
            *tb.symmetrize(times, C_vals, 0.0, lambda x: np.conj(x)),
            bc_type="natural",
            extrapolate=False
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

        times, phi = solve_pseudo_dyson(
            self.reservoir.g_less_t_fun(Q),
            self.reservoir.g_grea_t_fun(Q),
            t,
            sign * self.PP.capac_inv,
            self.AP.nr_pts_phi(t),
            method=self.AP.method,
        )
        return phi[-1]
