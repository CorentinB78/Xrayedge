import toolbox as tb
import numpy as np
from scipy import interpolate, integrate
from copy import copy
from functools import lru_cache


class Reservoir:
    """
    Abstract class for a generic reservoir.
    """

    def __init__(self):
        pass

    def g_less_t(self, Q, orb_a=0, orb_b=0):
        """
        Lesser GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    def g_grea_t(self, Q, orb_a=0, orb_b=0):
        """
        Greater GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    @lru_cache
    def g_less_t_fun(self, Q):
        """
        Lesser GF in times of QPC's central site.

        Returns a (cached) function
        """

        @lru_cache
        def func(orb_a, orb_b):
            times, g_less_t = self.g_less_t(Q, orb_a=orb_a, orb_b=orb_b)
            return interpolate.CubicSpline(times, g_less_t, extrapolate=False)

        return func

    @lru_cache
    def g_grea_t_fun(self, Q):
        """
        Greater GF in times of QPC's central site.

        Returns a (cached) function
        """

        @lru_cache
        def func(orb_a, orb_b):
            times, g_grea_t = self.g_grea_t(Q, orb_a=orb_a, orb_b=orb_b)
            return interpolate.CubicSpline(times, g_grea_t, extrapolate=False)

        return func


class OneDChainBetweenTwoLeads(Reservoir):
    """
    Abstract class for a reservoir made of a 1D central chain coupled at each end to a lead. Spinless fermions only.
    """

    def __init__(self, physics_params, nr_samples_fft, w_max):
        """
        Arguments:
            physics_params -- anobject containing parameters 'beta', 'bias_res', 'hamiltonian_res', 'orbitals', 'couplings'.

        Keyword Arguments:
            nr_samples_fft -- number of points for FFT
            w_max -- max frequency for FFT
        """
        super().__init__()
        self.PP = copy(physics_params)

        assert self.PP.hamiltonian_res.ndim == 2

        self.w_max = w_max
        self.N_fft = nr_samples_fft

    def delta_leads_R_left(self, w_array):
        """
        Retarded hybridization function in frequencies for left lead.
        """
        raise NotImplementedError

    def delta_leads_R_right(self, w_array):
        """
        Retarded hybridization function in frequencies for right lead.
        """
        raise NotImplementedError

    def delta_leads_K_left(self, w_array):
        """
        Keldysh hybridization function in frequencies for left lead.
        """
        w_array = np.atleast_1d(w_array)
        return (
            -(2 * tb.fermi(w_array, 0.5 * self.PP.bias_res, self.PP.beta) - 1.0)
            * 1j
            * np.imag(self.delta_leads_R_left(w_array))
        )

    def delta_leads_K_right(self, w_array):
        """
        Keldysh hybridization function in frequencies for right lead.
        """
        w_array = np.atleast_1d(w_array)
        return (
            -(2 * tb.fermi(w_array, -0.5 * self.PP.bias_res, self.PP.beta) - 1.0)
            * 1j
            * np.imag(self.delta_leads_R_right(w_array))
        )

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies.

        Returns a 3D array of shape (frequencies, space, space)
        """
        w_array = np.atleast_1d(w_array)
        N = len(self.PP.hamiltonian_res)
        couplings = np.zeros(N)
        for orb, V in zip(self.PP.orbitals, self.PP.couplings):
            couplings[orb] = V

        mat = np.empty((len(w_array), N, N), dtype=complex)
        mat[...] = (
            w_array[:, None, None] * np.eye(N)[None, :, :]
            - self.PP.hamiltonian_res[None, :, :]
        )
        mat[...] -= Q * np.diag(couplings)[None, :, :]
        mat[:, 0, 0] -= self.delta_leads_R_left(w_array)
        mat[:, -1, -1] -= self.delta_leads_R_right(w_array)

        return np.linalg.inv(mat)

    def g_less(self, w_array, Q):
        """
        Lesser GF in frequencies.

        Returns a 3D array of shape (frequencies, space, space)
        """
        w_array = np.atleast_1d(w_array)
        GR = self.g_reta(w_array, Q)
        GA = np.conj(GR).swapaxes(1, 2)

        left = GR[:, :, 0:1] * GA[:, 0:1, :]
        left *= self.delta_leads_K_left(w_array)[:, None, None] - 1j * np.imag(
            self.delta_leads_R_left(w_array)[:, None, None]
        )

        right = GR[:, :, -1:] * GA[:, -1:, :]
        right *= self.delta_leads_K_right(w_array)[:, None, None] - 1j * np.imag(
            self.delta_leads_R_right(w_array)[:, None, None]
        )

        return left + right

    def g_grea(self, w_array, Q):
        """
        Greater GF in frequencies.

        Returns a 3D array of shape (frequencies, space, space)
        """
        w_array = np.atleast_1d(w_array)
        GR = self.g_reta(w_array, Q)
        GA = np.conj(GR).swapaxes(1, 2)

        left = GR[:, :, 0:1] * GA[:, 0:1, :]
        left *= self.delta_leads_K_left(w_array)[:, None, None] + 1j * np.imag(
            self.delta_leads_R_left(w_array)[:, None, None]
        )

        right = GR[:, :, -1:] * GA[:, -1:, :]
        right *= self.delta_leads_K_right(w_array)[:, None, None] + 1j * np.imag(
            self.delta_leads_R_right(w_array)[:, None, None]
        )

        return left + right

    def g_less_t(self, Q, orb_a, orb_b):
        """
        Lesser GF in times

        Returns (times, vals) a pair of 1D arrays
        """
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_less = self.g_less(w, Q=Q)[:, orb_a, orb_b]

        times, g_less_t = tb.inv_fourier_transform(w, g_less, axis=0)
        return times, g_less_t

    def g_grea_t(self, Q, orb_a, orb_b):
        """
        Greater GF in times

        Returns (times, vals) a pair of 1D arrays
        """
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_grea = self.g_grea(w, Q=Q)[:, orb_a, orb_b]

        times, g_grea_t = tb.inv_fourier_transform(w, g_grea, axis=0)
        return times, g_grea_t

    def occupation(self, Q):
        w, dw = np.linspace(-self.w_max, self.w_max, self.N_fft, retstep=True)
        g_less = self.g_less(w, Q=Q).diagonal(0, 1, 2)

        return integrate.simpson(y=g_less.imag, dx=dw, axis=0) / (2 * np.pi)

    def transmission(self, w_array, Q):
        w_array = np.atleast_1d(w_array)
        GR = self.g_reta(w_array, Q)[:, 0, -1]
        return (
            self.delta_leads_R_left(w_array).imag
            * self.delta_leads_R_right(w_array).imag
            * np.abs(GR) ** 2
        )


class QuantumDot(OneDChainBetweenTwoLeads):
    """
    Quantum Dot. A type of reservoir with a central site coupled to two baths of infintie bandwidth and with a chemical potential difference.

    Provides real time Green functions at the central site for different charge offsets Q.
    """

    def __init__(self, physics_params, nr_samples_fft, w_max):
        """
        Arguments:
            physics_params -- an object with parameters `D_res`, `eps_res`, `bias_res`, `beta`, `orbitals` and `couplings`.

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT (default: {None} which auto determine an optimal value)
            w_max -- max frequency for FFT (default: {None} which auto determine an optimal value)
        """
        PP = copy(physics_params)
        PP.hamiltonian_res = np.array([[PP.eps_res]])
        del PP.eps_res
        super().__init__(PP, nr_samples_fft, w_max)

    def delta_leads_R_left(self, w_array):
        """
        Retarded hybridization function in frequencies for left lead.
        """
        w_array = np.atleast_1d(w_array)
        return -1j * self.PP.D_res / 2.0 * np.ones_like(w_array)

    def delta_leads_R_right(self, w_array):
        """
        Retarded hybridization function in frequencies for right lead.
        """
        w_array = np.atleast_1d(w_array)
        return -1j * self.PP.D_res / 2.0 * np.ones_like(w_array)


class QPC(OneDChainBetweenTwoLeads):
    """
    Quantum Point Contact. A type of reservoir with a central site coupled to two semicircular baths and with a chemical potential difference.

    Provides real time Green functions at the central site for different charge offsets Q.
    """

    def __init__(self, physics_params, nr_samples_fft, w_max):
        """
        Arguments:
            physics_params -- an object with parameters `D_res`, `eps_res`, `mu_res`, `eta_res`, `bias_res`, `beta`, `couplings` and `orbitals`.

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT
            w_max -- max frequency for FFT
        """
        PP = copy(physics_params)
        PP.hamiltonian_res = np.array(
            [[PP.eps_res + PP.D_res - PP.mu_res - PP.eta_res * 1j]]
        )
        del PP.eps_res
        super().__init__(PP, nr_samples_fft, w_max)

    def delta_leads_R_left(self, w_array):
        """
        Retarded hybridization function in frequencies for left lead.
        """
        w_array = np.atleast_1d(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = tb.semicirc_retarded_gf(t)
        return t**2 * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)

    def delta_leads_R_right(self, w_array):
        """
        Retarded hybridization function in frequencies for right lead.
        """
        w_array = np.atleast_1d(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = tb.semicirc_retarded_gf(t)
        return t**2 * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)
