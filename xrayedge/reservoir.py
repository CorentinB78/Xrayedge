import numpy as np
from scipy import interpolate, integrate
from copy import copy
from functools import lru_cache
from .fourier import inv_fourier_transform


def fermi(omegas, mu, beta):
    """
    Fermi function

    1 / (1 + e^{-beta (omegas - mu)})

    Entirely vectorized, supports infinite beta.
    """
    x = beta * (omegas - mu)
    ### for case beta=inf and omega=mu:
    x = np.nan_to_num(x, copy=False, nan=-1.0, posinf=+np.inf, neginf=-np.inf)
    return 0.5 * (1.0 + np.tanh(-x * 0.5))


def semicirc_retarded_gf(hopping):
    """
    Creates a callable Green function with semi-circular density of states.

    Arguments:
        hopping -- float

    Returns:
        vectorized callable z -> G(z), with z potenitally complex
    """
    g = abs(hopping)

    def gf(w):
        if np.real(w) <= -2 * g:
            return (w + np.sqrt(w**2 - 4 * g**2)) / (2 * g**2)
        elif abs(np.real(w)) < 2 * g:
            return (w - 1j * np.sqrt(4 * g**2 - w**2)) / (2 * g**2)
        else:
            return (w - np.sqrt(w**2 - 4 * g**2)) / (2 * g**2)

    return np.vectorize(gf, otypes=[complex])


class Reservoir:
    """
    Abstract class for a generic reservoir.
    """

    def __init__(self):
        pass

    def g_less_t(self, Q):
        """
        Lesser GF in times

        Returns:
            times -- 1D array
            values -- 3D array of shape (times, space, space)
        """
        raise NotImplementedError

    def g_grea_t(self, Q):
        """
        Greater GF in times

        Returns:
            times -- 1D array
            values -- 3D array of shape (times, space, space)
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
            times, g_less_t = self.g_less_t(Q)
            return interpolate.CubicSpline(
                times, g_less_t[:, orb_a, orb_b], extrapolate=False
            )

        return func

    @lru_cache
    def g_grea_t_fun(self, Q):
        """
        Greater GF in times of QPC's central site.

        Returns a (cached) function
        """

        @lru_cache
        def func(orb_a, orb_b):
            times, g_grea_t = self.g_grea_t(Q)
            return interpolate.CubicSpline(
                times, g_grea_t[:, orb_a, orb_b], extrapolate=False
            )

        return func


class OneDChainBetweenTwoLeads(Reservoir):
    """
    Abstract class for a reservoir made of a 1D central chain coupled at each end to a lead. Spinless fermions only.
    """

    def __init__(self, physics_params, nr_samples_fft, w_max, verbose=False):
        """
        Arguments:
            physics_params -- an object containing parameters 'beta', 'bias_res', 'hamiltonian_res', 'orbitals', 'couplings' and 'eta_res'.

        Keyword Arguments:
            nr_samples_fft -- number of points for FFT
            w_max -- max frequency for FFT
        """
        super().__init__()
        self.PP = copy(physics_params)

        assert self.PP.hamiltonian_res.ndim == 2
        assert len(self.PP.orbitals) == len(self.PP.couplings)
        assert len(self.PP.couplings) <= len(self.PP.hamiltonian_res)

        self.w_max = w_max
        self.N_fft = nr_samples_fft
        self.verbose = verbose

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
            -2j
            * (2 * fermi(w_array, 0.5 * self.PP.bias_res, self.PP.beta) - 1.0)
            * np.imag(self.delta_leads_R_left(w_array))
        )

    def delta_leads_K_right(self, w_array):
        """
        Keldysh hybridization function in frequencies for right lead.
        """
        w_array = np.atleast_1d(w_array)
        return (
            -2j
            * (2 * fermi(w_array, -0.5 * self.PP.bias_res, self.PP.beta) - 1.0)
            * np.imag(self.delta_leads_R_right(w_array))
        )

    def delta_less_center(self, w_array):
        w_array = np.atleast_1d(w_array)
        return 2j * self.PP.eta_res * fermi(w_array, 0.0, self.PP.beta)

    def delta_grea_center(self, w_array):
        w_array = np.atleast_1d(w_array)
        return -2j * self.PP.eta_res * fermi(w_array, 0.0, -self.PP.beta)

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
            np.eye(N)[None, :, :] * (w_array + 1.0j * self.PP.eta_res)[:, None, None]
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
        left *= 0.5 * self.delta_leads_K_left(w_array)[:, None, None] - 1j * np.imag(
            self.delta_leads_R_left(w_array)[:, None, None]
        )

        center = np.matmul(GR, GA) * self.delta_less_center(w_array)[:, None, None]

        right = GR[:, :, -1:] * GA[:, -1:, :]
        right *= 0.5 * self.delta_leads_K_right(w_array)[:, None, None] - 1j * np.imag(
            self.delta_leads_R_right(w_array)[:, None, None]
        )

        return left + center + right

    def g_grea(self, w_array, Q):
        """
        Greater GF in frequencies.

        Returns a 3D array of shape (frequencies, space, space)
        """
        w_array = np.atleast_1d(w_array)
        GR = self.g_reta(w_array, Q)
        GA = np.conj(GR).swapaxes(1, 2)

        left = GR[:, :, 0:1] * GA[:, 0:1, :]
        left *= 0.5 * self.delta_leads_K_left(w_array)[:, None, None] + 1j * np.imag(
            self.delta_leads_R_left(w_array)[:, None, None]
        )

        center = np.matmul(GR, GA) * self.delta_grea_center(w_array)[:, None, None]

        right = GR[:, :, -1:] * GA[:, -1:, :]
        right *= 0.5 * self.delta_leads_K_right(w_array)[:, None, None] + 1j * np.imag(
            self.delta_leads_R_right(w_array)[:, None, None]
        )

        return left + center + right

    @lru_cache
    def g_less_t(self, Q):
        """
        Lesser GF in times

        Returns:
            times -- 1D array
            values -- 3D array of shape (times, space, space)
        """
        if self.verbose:
            print("[Xray/Reservoir] computing g_less_t...", flush=True)

        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_less = self.g_less(w, Q=Q)

        times, g_less_t = inv_fourier_transform(w, g_less, axis=0)
        return times, g_less_t

    @lru_cache
    def g_grea_t(self, Q):
        """
        Greater GF in times

        Returns:
            times -- 1D array
            values -- 3D array of shape (times, space, space)
        """
        if self.verbose:
            print("[Xray/Reservoir] computing g_grea_t...", flush=True)

        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_grea = self.g_grea(w, Q=Q)

        times, g_grea_t = inv_fourier_transform(w, g_grea, axis=0)
        return times, g_grea_t

    def occupation(self, Q):
        w, dw = np.linspace(-self.w_max, self.w_max, self.N_fft, retstep=True)
        g_less = self.g_less(w, Q=Q).diagonal(0, 1, 2)

        return integrate.simpson(y=g_less.imag, dx=dw, axis=0) / (2 * np.pi)

    def transmission(self, w_array, Q):
        w_array = np.atleast_1d(w_array)
        GR = self.g_reta(w_array, Q)[:, 0, -1]
        return (
            4
            * self.delta_leads_R_left(w_array).imag
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
            physics_params -- an object with parameters `D_res`, `eps_res`, `bias_res`, `beta`, `orbitals`, `couplings` and `eta_res`.

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
        PP.hamiltonian_res = np.array([[PP.eps_res + PP.D_res - PP.mu_res]])
        del PP.eps_res
        super().__init__(PP, nr_samples_fft, w_max)

    def delta_leads_R_left(self, w_array):
        """
        Retarded hybridization function in frequencies for left lead.
        """
        w_array = np.atleast_1d(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = semicirc_retarded_gf(t)
        return t**2 * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)

    def delta_leads_R_right(self, w_array):
        """
        Retarded hybridization function in frequencies for right lead.
        """
        w_array = np.atleast_1d(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = semicirc_retarded_gf(t)
        return t**2 * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)


class ExtendedQPC(OneDChainBetweenTwoLeads):
    def __init__(self, physics_params, nr_samples_fft, w_max):
        """
        Arguments:
            physics_params -- an object with parameters `D_res`, `eps_res`, `mu_res`, `eta_res`, `bias_res`, `beta`, `couplings` and `orbitals`.

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT
            w_max -- max frequency for FFT
        """
        PP = copy(physics_params)
        N = len(PP.eps_res)
        H = np.zeros((N, N), dtype=complex)

        for i in range(N):
            H[i, i] = PP.eps_res[i] + PP.D_res - PP.mu_res

        t = PP.D_res / 2.0  # hopping
        for i in range(N - 1):
            H[i, i + 1] = -t
            H[i + 1, i] = -t

        PP.hamiltonian_res = H
        del PP.eps_res
        super().__init__(PP, nr_samples_fft, w_max)

    def delta_leads_R_left(self, w_array):
        """
        Retarded hybridization function in frequencies for left lead.
        """
        w_array = np.atleast_1d(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = semicirc_retarded_gf(t)
        return t**2 * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)

    def delta_leads_R_right(self, w_array):
        """
        Retarded hybridization function in frequencies for right lead.
        """
        w_array = np.atleast_1d(w_array)
        t = self.PP.D_res / 2.0  # hopping
        sc_gf = semicirc_retarded_gf(t)
        return t**2 * sc_gf(w_array + self.PP.mu_res - 2.0 * t + self.PP.eta_res * 1j)
