import toolbox as tb
import numpy as np
from scipy import interpolate
from copy import copy


class Reservoir:
    def __init__(self):
        """
        Abstract class for a generic reservoir.
        """
        self.N = 3  # nr of different charge states affecting the QPC

        self._cache_g_less_t = [None] * self.N
        self._cache_g_grea_t = [None] * self.N

    def g_less_t(self, Q):
        """
        Lesser GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    def g_grea_t(self, Q):
        """
        Greater GF in times of contact site.

        Returns times, values
        """
        raise NotImplementedError

    def g_less_t_fun(self, Q):
        """
        Lesser GF in times of QPC's central site.

        Returns a (cached) function
        """
        if self._cache_g_less_t[Q] is None:
            times, g_less_t = self.g_less_t(Q)
            self._cache_g_less_t[Q] = interpolate.CubicSpline(times, g_less_t)

        return self._cache_g_less_t[Q]

    def g_grea_t_fun(self, Q):
        """
        Greater GF in times of QPC's central site.

        Returns a (cached) function
        """
        if self._cache_g_grea_t[Q] is None:
            times, g_grea_t = self.g_grea_t(Q)
            self._cache_g_grea_t[Q] = interpolate.CubicSpline(times, g_grea_t)

        return self._cache_g_grea_t[Q]


class QPC(Reservoir):
    def __init__(self, physics_params, nr_samples_fft=None, w_max=None):
        """
        Quantum Point Contact. A type of reservoir with a central site coupled to two baths with a chemical potential difference.

        Provides real time Green functions at the central site for different charge offsets Q.

        Arguments:
            physics_params -- a `PhysicsParameters` instance

        Keyword Arguments:
            nr_samples_fft -- number of grid points for FFT (default: {None} which auto determine an optimal value)
            w_max -- max frequency for FFT (default: {None} which auto determine an optimal value)
        """
        super().__init__()
        self.PP = copy(physics_params)

        spreads = np.array([self.PP.Gamma, 1.0 / self.PP.beta])
        centers = np.array(
            [
                self.PP.eps_c,
                self.PP.mu_c + 0.5 * self.PP.bias,
                self.PP.mu_c - 0.5 * self.PP.bias,
            ]
        )

        if w_max is None:
            w_max = np.max(np.abs(centers[:, None] + spreads[None, :]))
            w_max = 100 * max(
                w_max, np.max(np.abs(centers[:, None] - spreads[None, :]))
            )
        self.w_max = w_max

        if nr_samples_fft is None:
            dw = np.min(spreads) / 100.0
            self.N_fft = int(2 * w_max / dw + 0.5)
        else:
            self.N_fft = nr_samples_fft

        if self.N_fft >= int(1e6):
            print(
                f"/!\ [Reservoir] FFT requires {self.N_fft} grid points. Capped to 10^6."
            )
            self.N_fft = int(1e6)

    def delta_leads_R(self, w_array):
        """
        Retarded hybridization function in frequencies for left and right leads (together) of QPC.
        """
        return -1j * self.PP.Gamma * np.ones_like(w_array)

    def delta_leads_K(self, w_array):
        """
        Keldysh hybridization function in frequencies for left and right leads (together) of QPC.
        """
        return (
            -2j
            * (
                tb.fermi(w_array, self.PP.mu_c + 0.5 * self.PP.bias, self.PP.beta)
                + tb.fermi(w_array, self.PP.mu_c - 0.5 * self.PP.bias, self.PP.beta)
                - 1.0
            )
            * np.imag(self.delta_leads_R(w_array))
        )

    def g_reta(self, w_array, Q):
        """
        Retarded GF in frequencies of QPC's central site.
        """
        return 1.0 / (
            w_array
            - self.PP.eps_c
            - Q * self.PP.capac_inv
            - self.delta_leads_R(w_array)
        )

    def g_keld(self, w_array, Q):
        """
        Keldysh GF in frequencies of QPC's central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * self.delta_leads_K(w_array)

    def g_less(self, w_array, Q):
        """
        Lesser GF in frequencies of QPC's central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * (
            0.5 * self.delta_leads_K(w_array)
            - 1.0j * np.imag(self.delta_leads_R(w_array))
        )

    def g_grea(self, w_array, Q):
        """
        Greater GF in frequencies of QPC's central site.
        """
        return np.abs(self.g_reta(w_array, Q)) ** 2 * (
            0.5 * self.delta_leads_K(w_array)
            + 1.0j * np.imag(self.delta_leads_R(w_array))
        )

    def g_less_t(self, Q):
        """
        Lesser GF in times of contact site.
        """
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_less = self.g_less(w, Q=Q)

        times, g_less_t = tb.inv_fourier_transform(w, g_less)
        return times, g_less_t

    def g_grea_t(self, Q):
        """
        Greater GF in times of contact site.
        """
        w = np.linspace(-self.w_max, self.w_max, self.N_fft)
        g_grea = self.g_grea(w, Q=Q)

        times, g_grea_t = tb.inv_fourier_transform(w, g_grea)
        return times, g_grea_t
